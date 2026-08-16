"""
Merges the LoRA adapter produced by train_dpo.py into the base model weights,
producing a single full model directory that SGLang can serve directly with
`--model-path`. Run this at the end of the Colab session, then download the
merged directory (outputs/dpo_model_merged) to your local machine.

Usage:
    python src/merge_lora.py
"""
import os

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import CFG


def main():
    adapter_path = os.path.join(CFG.output_dir, "dpo_model")
    merged_path = os.path.join(CFG.output_dir, "dpo_model_merged")

    if not os.path.exists(adapter_path):
        raise FileNotFoundError(
            f"{adapter_path} not found. Run train_dpo.py first."
        )

    print(f"Loading base model: {CFG.model_name}")
    base = AutoModelForCausalLM.from_pretrained(
        CFG.model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )
    tokenizer = AutoTokenizer.from_pretrained(CFG.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading LoRA adapter: {adapter_path}")
    model = PeftModel.from_pretrained(base, adapter_path)

    print("Merging adapter into base weights...")
    merged = model.merge_and_unload()

    os.makedirs(merged_path, exist_ok=True)
    merged.save_pretrained(merged_path, safe_serialization=True)
    tokenizer.save_pretrained(merged_path)

    print(f"Merged model saved to {merged_path}")
    print("Download this folder to your local machine, then serve it with:")
    print(f"  python -m sglang.launch_server --model-path {merged_path} --port 30000")


if __name__ == "__main__":
    main()
