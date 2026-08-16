"""
Runs base, SFT, and DPO-tuned models against both eval sets and reports
ROUGE-L, entity-F1, fact-consistency, and how often the output was malformed.

Deliberately doesn't use SGLang's constrained decoding here -- the point is
to see how much the *training itself* fixed the malformed-output problem,
separate from the structural guarantee SGLang gives you at serving time.

    python src/evaluate.py --n-examples 50
"""
import argparse
import json
import os

import torch
from rouge_score import rouge_scorer
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from config import CFG
from data import load_split, load_aci_bench_eval
from reward import entity_f1, fact_consistency
from schema import is_well_formed

PROMPT_TEMPLATE = """You are a clinical scribe. Summarize the following doctor-patient \
dialogue into a structured clinical note as JSON with keys: chief_complaint, \
history_of_present_illness, assessment, plan, medications_mentioned. \
Only include facts stated in the dialogue.

Dialogue:
{dialogue}

JSON note:"""


def load_model(path_or_name: str, is_lora_adapter: bool = False):
    tokenizer = AutoTokenizer.from_pretrained(
        path_or_name if not is_lora_adapter else CFG.model_name
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = "cuda" if torch.cuda.is_available() else "cpu"
    base = AutoModelForCausalLM.from_pretrained(
        CFG.model_name if is_lora_adapter else path_or_name,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
    ).to(device)

    if is_lora_adapter:
        model = PeftModel.from_pretrained(base, path_or_name).to(device)
    else:
        model = base
    model.eval()
    return model, tokenizer, device


def generate(model, tokenizer, device, dialogue: str) -> str:
    prompt = PROMPT_TEMPLATE.format(dialogue=dialogue)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                        max_length=CFG.max_seq_len).to(device)
    out = model.generate(
        **inputs, max_new_tokens=300, do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def evaluate_model(model, tokenizer, device, examples, label: str):
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    rouge_scores, ent_f1s, fact_scores, malformed_count = [], [], [], 0

    for ex in examples:
        pred = generate(model, tokenizer, device, ex["dialogue"])
        if not is_well_formed(pred):
            malformed_count += 1
        rouge_scores.append(scorer.score(ex["summary"], pred)["rougeL"].fmeasure)
        try:
            ent_f1s.append(entity_f1(ex["dialogue"], pred))
            fact_scores.append(fact_consistency(ex["dialogue"], pred))
        except Exception as e:  # noqa: BLE001
            print(f"Scoring failed for one example: {e}")

    n = len(examples)
    results = {
        "label": label,
        "rougeL": sum(rouge_scores) / len(rouge_scores) if rouge_scores else 0.0,
        "entity_f1": sum(ent_f1s) / len(ent_f1s) if ent_f1s else 0.0,
        "fact_consistency": sum(fact_scores) / len(fact_scores) if fact_scores else 0.0,
        "malformed_rate": malformed_count / n if n else 0.0,
        "n_examples": n,
    }
    return results


def main(n_examples: int):
    test_mts = load_split("test")[:n_examples]
    test_aci = load_aci_bench_eval()  # small (66 examples), run in full -- the harder/independent set

    eval_sets = {"mts_dialog_test": test_mts, "aci_bench_external": test_aci}

    print("Evaluating base model...")
    base_model, tokenizer, device = load_model(CFG.model_name, is_lora_adapter=False)
    base_results = {name: evaluate_model(base_model, tokenizer, device, ex, f"base/{name}")
                     for name, ex in eval_sets.items()}
    del base_model
    torch.cuda.empty_cache()

    sft_path = os.path.join(CFG.output_dir, "sft_model")
    sft_results = None
    if os.path.exists(sft_path):
        print("Evaluating SFT baseline model...")
        sft_model, tokenizer, device = load_model(sft_path, is_lora_adapter=True)
        sft_results = {name: evaluate_model(sft_model, tokenizer, device, ex, f"sft_baseline/{name}")
                        for name, ex in eval_sets.items()}
        del sft_model
        torch.cuda.empty_cache()
    else:
        print(f"No SFT model found at {sft_path}, skipping. Run train_sft.py first "
              f"if you want the full base -> SFT -> DPO comparison (matches the original "
              f"resume bullet's 'vs. supervised baseline' framing).")

    dpo_path = os.path.join(CFG.output_dir, "dpo_model")
    dpo_results = None
    if os.path.exists(dpo_path):
        print("Evaluating DPO-tuned model...")
        dpo_model, tokenizer, device = load_model(dpo_path, is_lora_adapter=True)
        dpo_results = {name: evaluate_model(dpo_model, tokenizer, device, ex, f"dpo_tuned/{name}")
                        for name, ex in eval_sets.items()}
    else:
        print(f"No DPO model found at {dpo_path}, skipping. Run train_dpo.py first.")

    out = {"base": base_results, "sft_baseline": sft_results, "dpo_tuned": dpo_results}
    out_path = os.path.join(CFG.output_dir, "eval_results.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    print(json.dumps(out, indent=2))
    print(f"\nSaved to {out_path}")
    print(
        "\nNote: 'mts_dialog_test' is a held-out split of the training distribution; "
        "'aci_bench_external' is a fully independent, harder, full-note dataset never "
        "seen during training or RAG indexing -- treat it as the more meaningful number. "
        "For the 'correlation with expert judgment' metric, run src/human_eval.py separately "
        "-- it needs an actual human to fill in ratings, so it can't be part of this script."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-examples", type=int, default=50)
    args = parser.parse_args()
    main(args.n_examples)
