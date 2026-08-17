"""
Plain supervised fine-tuning (LoRA) on (dialogue -> summary) pairs.
This is the missing middle rung: base (zero-shot) -> SFT (this script) -> DPO (train_dpo.py).
Needed so evaluate.py's comparison is actually "vs. supervised baseline", not just "vs. base".
"""
import os

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling

from config import CFG
from data import load_split, row_to_clinical_note_json

PROMPT_TEMPLATE = """You are a clinical scribe. Summarize the following doctor-patient \
dialogue into a structured clinical note as JSON with keys: chief_complaint, \
history_of_present_illness, assessment, plan, medications_mentioned. \
Only include facts stated in the dialogue.

Dialogue:
{dialogue}

JSON note:"""


def build_sft_examples(tokenizer):
    train = load_split("train")
    texts = []
    for ex in train:
        prompt = PROMPT_TEMPLATE.format(dialogue=ex["dialogue"])
        target = row_to_clinical_note_json(ex.get("section_header", ""), ex["summary"])
        full = prompt + " " + target + tokenizer.eos_token
        texts.append(full)
    return Dataset.from_dict({"text": texts})


def tokenize_fn(tokenizer):
    def _fn(batch):
        return tokenizer(batch["text"], truncation=True, max_length=CFG.max_seq_len,
                          padding="max_length")
    return _fn


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(CFG.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        CFG.model_name,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
    )

    peft_config = LoraConfig(
        r=CFG.lora_r, lora_alpha=CFG.lora_alpha, lora_dropout=CFG.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)

    ds = build_sft_examples(tokenizer)
    ds = ds.map(tokenize_fn(tokenizer), batched=True, remove_columns=["text"])

    sft_out_dir = os.path.join(CFG.output_dir, "sft_model")
    args = TrainingArguments(
        output_dir=sft_out_dir,
        num_train_epochs=CFG.num_train_epochs,
        per_device_train_batch_size=CFG.per_device_train_batch_size,
        gradient_accumulation_steps=CFG.gradient_accumulation_steps,
        learning_rate=CFG.learning_rate,
        logging_steps=10,
        save_strategy="epoch",
        bf16=(device == "cuda"),
        report_to=[],
    )
    collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    trainer = Trainer(model=model, args=args, train_dataset=ds, data_collator=collator)
    trainer.train()
    trainer.save_model(sft_out_dir)
    tokenizer.save_pretrained(sft_out_dir)
    print(f"Saved SFT baseline model to {sft_out_dir}")


if __name__ == "__main__":
    main()
