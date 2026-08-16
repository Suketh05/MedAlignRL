"""
LoRA + DPO on the preference pairs from preference_pairs.py, via TRL's
DPOTrainer. Picks up the SFT checkpoint automatically if train_sft.py has
been run first; otherwise starts DPO straight from the base model (still
works, just skips the middle rung of the base -> SFT -> DPO comparison).
"""
import json
import os

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer

from config import CFG


def load_preference_dataset():
    path = os.path.join(CFG.data_dir, "preference_pairs.jsonl")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Run preference_pairs.py first."
        )
    rows = []
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            rows.append({
                "prompt": d["prompt"],
                "chosen": d["chosen"],
                "rejected": d["rejected"],
            })
    return Dataset.from_list(rows)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(CFG.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    sft_path = os.path.join(CFG.output_dir, "sft_model")
    if os.path.exists(sft_path):
        print(f"Starting DPO from SFT checkpoint: {sft_path}")
        from peft import PeftModel
        base = AutoModelForCausalLM.from_pretrained(
            CFG.model_name,
            torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        )
        model = PeftModel.from_pretrained(base, sft_path).merge_and_unload()
    else:
        print("No SFT checkpoint found -- starting DPO from the base model directly. "
              "Run train_sft.py first for the full base -> SFT -> DPO comparison.")
        model = AutoModelForCausalLM.from_pretrained(
            CFG.model_name,
            torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        )

    peft_config = LoraConfig(
        r=CFG.lora_r,
        lora_alpha=CFG.lora_alpha,
        lora_dropout=CFG.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",
    )

    ds = load_preference_dataset()
    print(f"Loaded {len(ds)} preference pairs")
    split = ds.train_test_split(test_size=0.1, seed=CFG.seed)

    dpo_out_dir = os.path.join(CFG.output_dir, "dpo_model")

    training_args = DPOConfig(
        output_dir=dpo_out_dir,
        beta=CFG.dpo_beta,
        learning_rate=CFG.learning_rate,
        num_train_epochs=CFG.num_train_epochs,
        per_device_train_batch_size=CFG.per_device_train_batch_size,
        gradient_accumulation_steps=CFG.gradient_accumulation_steps,
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch",
        bf16=(device == "cuda"),
        report_to=[],
        max_prompt_length=1024,
        max_length=1536,
    )

    trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=split["train"],
        eval_dataset=split["test"],
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    trainer.train()
    trainer.save_model(dpo_out_dir)
    tokenizer.save_pretrained(dpo_out_dir)
    print(f"Saved DPO-tuned model to {dpo_out_dir}")


if __name__ == "__main__":
    main()
