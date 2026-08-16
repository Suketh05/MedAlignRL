"""
Every automated metric so far (ROUGE-L, entity-F1, NLI fact-consistency) is a
proxy for what a person would actually think. This is the check on that: sample
some outputs, get a human to rate them 1-5, then see whether the automated
reward and plain ROUGE actually track that rating -- and which one tracks it
better.

    python src/human_eval.py --mode generate --model-path ../outputs/dpo_model_merged --n 30
    # fill in outputs/human_eval_sheet.csv by hand, then:
    python src/human_eval.py --mode score
"""
import argparse
import csv
import os

import torch
from rouge_score import rouge_scorer
from scipy.stats import pearsonr
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import CFG
from data import load_split
from reward import composite_reward

SHEET_PATH = os.path.join(CFG.output_dir, "human_eval_sheet.csv")

PROMPT_TEMPLATE = """You are a clinical scribe. Summarize the following doctor-patient \
dialogue into a structured clinical note as JSON with keys: chief_complaint, \
history_of_present_illness, assessment, plan, medications_mentioned. \
Only include facts stated in the dialogue.

Dialogue:
{dialogue}

JSON note:"""


def generate_sheet(model_path: str, n: int):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32
    ).to(device)
    model.eval()

    rouge_scorer_ = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    test = load_split("test")[:n]
    rows = []
    for ex in test:
        prompt = PROMPT_TEMPLATE.format(dialogue=ex["dialogue"])
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                            max_length=CFG.max_seq_len).to(device)
        out = model.generate(**inputs, max_new_tokens=300, do_sample=False,
                              pad_token_id=tokenizer.eos_token_id)
        pred = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:],
                                 skip_special_tokens=True).strip()
        auto_score = composite_reward(ex["dialogue"], pred)["total"]
        rouge_l = rouge_scorer_.score(ex["summary"], pred)["rougeL"].fmeasure
        rows.append({
            "id": ex["id"],
            "dialogue": ex["dialogue"],
            "generated_summary": pred,
            "composite_reward_score": round(auto_score, 4),
            "rouge_l_score": round(rouge_l, 4),
            "expert_rating": "",  # fill in 1-5: is every stated fact supported by the dialogue?
        })

    with open(SHEET_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {SHEET_PATH}")
    print("Open it, fill in 'expert_rating' (1-5, factual accuracy), then rerun with --mode score")


def score_sheet():
    if not os.path.exists(SHEET_PATH):
        raise FileNotFoundError(f"{SHEET_PATH} not found. Run --mode generate first.")

    composite_scores, rouge_scores, expert_scores = [], [], []
    with open(SHEET_PATH) as f:
        for row in csv.DictReader(f):
            if row["expert_rating"].strip() == "":
                continue
            composite_scores.append(float(row["composite_reward_score"]))
            rouge_scores.append(float(row["rouge_l_score"]))
            expert_scores.append(float(row["expert_rating"]))

    if len(composite_scores) < 3:
        print(f"Only {len(composite_scores)} rated rows found -- need at least a handful "
              f"of filled-in expert_rating values before correlation is meaningful.")
        return

    r_composite, p_composite = pearsonr(composite_scores, expert_scores)
    r_rouge, p_rouge = pearsonr(rouge_scores, expert_scores)

    print(f"Pearson correlation, composite reward vs. expert rating: r={r_composite:.3f}, p={p_composite:.4f}")
    print(f"Pearson correlation, ROUGE-L vs. expert rating:          r={r_rouge:.3f}, p={p_rouge:.4f}")

    if r_rouge != 0:
        pct_diff = (r_composite - r_rouge) / abs(r_rouge) * 100
        print(f"\nComposite reward correlates with expert judgment {pct_diff:+.1f}% "
              f"relative to ROUGE-L alone.")
    print(f"\nBased on {len(composite_scores)} rated examples.")
    print("\nHonesty note: report the reviewer's background alongside this number "
          "(e.g. 'rated by a med student', not just 'expert judgment') -- credibility "
          "of this correlation depends entirely on who did the rating. With n<30-ish, "
          "also report the p-value alongside r -- don't state the percentage difference "
          "without it, since small samples make correlation estimates noisy.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["generate", "score"], required=True)
    parser.add_argument("--model-path", type=str, default=None,
                         help="Path to a full (merged) model directory, required for --mode generate")
    parser.add_argument("--n", type=int, default=30)
    args = parser.parse_args()

    if args.mode == "generate":
        if not args.model_path:
            raise ValueError("--model-path is required for --mode generate")
        generate_sheet(args.model_path, args.n)
    else:
        score_sheet()
