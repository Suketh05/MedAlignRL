"""
Builds the DPO training set. For each training dialogue: sample a handful of
different summaries at temperature > 0, score each one with the reward
function, keep the best as "chosen" and the worst as "rejected".

This is the whole trick that makes this feasible without a PPO loop --
instead of a reward model sitting in the training loop scoring generations
as they happen, we run it once, offline, just to label pairs. Much cheaper,
and DPO can train directly on the labeled pairs afterward.
"""
import argparse
import json
import os

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import CFG
from data import load_split
from rag import Retriever
from reward import composite_reward

PROMPT_TEMPLATE = """You are a clinical scribe. Summarize the following doctor-patient \
dialogue into a structured clinical note with sections: Chief Complaint, \
History of Present Illness, Assessment, Plan. Only include facts stated in the dialogue.

Similar example dialogue and note (for reference only, do not copy facts from it):
---
Dialogue: {ex_dialogue}
Note: {ex_summary}
---

Now summarize this dialogue:
{dialogue}

Structured note:"""


def build_prompt(dialogue: str, retriever: Retriever) -> str:
    exemplars = retriever.retrieve(dialogue, k=1)
    ex = exemplars[0] if exemplars else {"dialogue": "", "summary": ""}
    return PROMPT_TEMPLATE.format(
        ex_dialogue=ex["dialogue"][:600],
        ex_summary=ex["summary"][:400],
        dialogue=dialogue,
    )


def sample_candidates(model, tokenizer, prompt: str, n: int, device) -> list:
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                        max_length=CFG.max_seq_len).to(device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=300,
        do_sample=True,
        temperature=CFG.sampling_temperature,
        top_p=0.95,
        num_return_sequences=n,
        pad_token_id=tokenizer.eos_token_id,
    )
    candidates = []
    for out in outputs:
        text = tokenizer.decode(out[inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        candidates.append(text.strip())
    return candidates


def main(n_candidates: int, n_examples: int):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    out_path = os.path.join(CFG.data_dir, "preference_pairs.jsonl")

    # Resume support: if this crashed or the Colab runtime dropped partway
    # through last time, don't start over -- skip whatever's already written
    # and pick up where it left off.
    already_done_ids = set()
    if os.path.exists(out_path):
        with open(out_path) as f:
            for line in f:
                try:
                    already_done_ids.add(json.loads(line)["id"])
                except Exception:  # noqa: BLE001
                    pass
        if already_done_ids:
            print(f"Found {len(already_done_ids)} pairs already written to {out_path} "
                  f"from a previous run -- resuming, not starting over.")

    tokenizer = AutoTokenizer.from_pretrained(CFG.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        CFG.model_name,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
    ).to(device)
    model.eval()

    retriever = Retriever()
    retriever.load()

    train = load_split("train")[:n_examples]
    remaining = [ex for ex in train if ex["id"] not in already_done_ids]
    if len(remaining) < len(train):
        print(f"{len(train) - len(remaining)} of {len(train)} already done, "
              f"{len(remaining)} left to process.")

    written = len(already_done_ids)

    # Append mode + flush after every write, so a disconnect only costs you
    # the one example being processed when it happens, not the whole run.
    with open(out_path, "a") as out_f:
        for ex in tqdm(remaining, desc="Building preference pairs"):
            prompt = build_prompt(ex["dialogue"], retriever)
            try:
                candidates = sample_candidates(model, tokenizer, prompt, n_candidates, device)
            except Exception as e:  # noqa: BLE001
                print(f"Generation failed for {ex['id']}: {e}")
                continue

            scored = []
            for cand in candidates:
                if not cand.strip():
                    continue
                try:
                    r = composite_reward(ex["dialogue"], cand)
                except Exception as e:  # noqa: BLE001
                    print(f"Reward scoring failed: {e}")
                    continue
                scored.append((cand, r["total"]))

            if len(scored) < 2:
                continue

            scored.sort(key=lambda x: x[1], reverse=True)
            chosen, chosen_score = scored[0]
            rejected, rejected_score = scored[-1]

            if chosen_score - rejected_score < 0.05:
                continue  # not enough signal to be a useful preference pair

            record = {
                "id": ex["id"],
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected,
                "chosen_score": chosen_score,
                "rejected_score": rejected_score,
            }
            out_f.write(json.dumps(record) + "\n")
            out_f.flush()
            os.fsync(out_f.fileno())
            written += 1

    print(f"Wrote {written} total preference pairs to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-candidates", type=int, default=CFG.n_candidates_per_example)
    parser.add_argument("--n-examples", type=int, default=CFG.n_preference_examples)
    args = parser.parse_args()
    main(args.n_candidates, args.n_examples)
