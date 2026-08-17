# MedAlign-RL: Clinical Summarization Agent

[![sanity-check](https://github.com/YOUR_USERNAME/MedAlignRL/actions/workflows/sanity-check.yml/badge.svg)](https://github.com/YOUR_USERNAME/MedAlignRL/actions/workflows/sanity-check.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A multi-agent clinical summarization pipeline that combines **RAG retrieval**,
**SGLang structured generation**, and **DPO-based RLHF** (driven by a composite
factuality reward) to turn doctor–patient dialogues into structured, low-hallucination
clinical note summaries with a small open LLM.

## Why this problem matters

Clinical documentation burden is one of the largest drivers of physician burnout —
clinicians spend roughly two hours writing notes for every hour of patient care.
LLM-based note generation is being built by every major EHR vendor and a wave of
startups (Abridge, Ambience, Suki, Nuance DAX), but the central failure mode is
**hallucinated clinical facts** — a summarization quality issue that is also a
patient-safety issue. This project treats factual grounding as a first-class
training signal instead of an afterthought bolted onto ROUGE.

## Architecture

```
 dialogue ──▶ [Retriever Agent] ──▶ k similar (dialogue, summary) exemplars
                     │
                     ▼
 dialogue + exemplars ──▶ [Drafter Agent] ──▶ draft summary (SGLang schema-constrained)
                     │
                     ▼
        [Verifier Agent] ── entity-F1 + NLI fact-divergence score
                     │
            score < threshold?
              │            │
             yes           no
              ▼            ▼
      [Refiner Agent]   final summary
              │
              └──▶ back to Verifier (max 1 retry)
```

Training loop (offline, before serving):

```
train dialogues ──▶ sample K candidate summaries per example (temperature sampling)
                 ──▶ score each with the composite reward (entity-F1, NLI divergence, discourse flow)
                 ──▶ build (chosen, rejected) preference pairs
                 ──▶ DPO fine-tune (LoRA) the base model on those pairs
```

DPO is used instead of full PPO: it needs no reward model in the training loop
(the reward model is only used once, offline, to *label* preference pairs), which
is what makes a same-day RLHF-style training run feasible on a single Colab GPU.
It is still a legitimate member of the RLHF family (RLHF-without-a-critic).

## Components

| File | Role |
|---|---|
| `src/config.py` | All hyperparameters / paths in one place |
| `src/data.py` | Loads MTS-Dialog (train) + ACI-Bench (held-out external eval) |
| `src/schema.py` | Pydantic schema for the structured note + SGLang regex/JSON constraints |
| `src/rag.py` | Sentence-embedding index over training exemplars; top-k retrieval |
| `src/reward.py` | Composite reward: entity-F1 (scispaCy/spaCy NER) + NLI fact-divergence + discourse flow |
| `src/preference_pairs.py` | Samples candidates, scores them, builds DPO preference dataset |
| `src/train_sft.py` | Plain supervised LoRA fine-tune — the "supervised baseline" rung between base and DPO |
| `src/train_dpo.py` | LoRA + DPO training with TRL |
| `src/merge_lora.py` | Merges a LoRA adapter into a full model for local SGLang serving |
| `src/agents.py` | The multi-agent pipeline (retriever → drafter → verifier → refiner) via SGLang |
| `src/evaluate.py` | ROUGE-L, entity-F1, malformed-output rate, fact-consistency — base vs. SFT vs. DPO-tuned, on both eval sets |
| `src/human_eval.py` | Generates a rating sheet for a human reviewer, then computes correlation between their 1–5 scores and the automated reward — this is what "correlation with expert judgment" actually requires |
| `notebooks/` | Six standalone, resumable Colab notebooks — see `notebooks/README.md` for the run order |

## Positioning: why build this, given adjacent work exists

Prompt-wrapper "AI clinical note" projects are common on GitHub, but almost all
of them call a hosted LLM API with a prompt and stop there — no retrieval
grounding, no factuality-aware reward, no RLHF-family training, no constrained
decoding. Separately, factuality-via-entailment-reward RL has been explored for
generic (news) summarization, and RLHF/DPO tooling itself is mature and
commoditized (TRL, RLHFlow, etc.). The specific combination here — RAG +
schema-constrained generation + a composite factuality reward (entity-F1 +
NLI) + DPO + a multi-agent verify/refine loop, applied end-to-end to clinical
dialogue summarization and open-sourced with real eval numbers — doesn't have
an obvious existing public equivalent. That combination, not any single piece
in isolation, is the point of this repo. It's presented as a faithful,
small-scale, transparent implementation of an underexplored technique, not a
claim of being first or state-of-the-art.

## Split workflow: train on Colab, run the agent pipeline locally

Training (data prep → RAG index → preference pairs → DPO) is GPU-heavy but
short-lived — Colab is the right place for it. The multi-agent SGLang pipeline
is meant to run continuously against a local server, which fits a local
machine better. The handoff:

1. **On Colab**: run through step 6 in the notebook (`merge_lora.py`), which
   merges the LoRA adapter into a full, directly-servable model.
2. **Download** `outputs/dpo_model_merged/` and `data/rag_index/` from Colab
   (the notebook's last cells zip and download both).
3. **On your local PC**: place both folders in the same relative paths inside
   your local clone, then:
   ```bash
   python -m sglang.launch_server --model-path /path/to/dpo_model_merged --port 30000 &
   python src/agents.py --dialogue "Doctor: ... Patient: ..." --endpoint http://localhost:30000
   ```
   If you'd rather not transfer the RAG index, rebuild it locally instead —
   it's CPU-only and fast: `python src/rag.py --build-index`.

## Datasets

Two open, no-credentialing datasets, used for different purposes — this is
the direct answer to "what's the proper dataset for this":

| Dataset | Role | Size | License | Why |
|---|---|---|---|---|
| **[MTS-Dialog](https://huggingface.co/datasets/har1/MTS_Dialogue-Clinical_Note)** | Training (RAG index, DPO preference pairs) | ~1,300 dialogue/section pairs | MIT | Enough volume to build a real retrieval index and sample diverse candidates for preference labeling |
| **[ACI-Bench](https://huggingface.co/datasets/mkieffer/ACI-Bench)** | Held-out **external** evaluation only, never trained on | 66 full dialogues + full structured notes (test1+test2+test3) | CC-BY-4.0 | Longer, harder, real full-visit conversations with properly sectioned notes (CHIEF COMPLAINT / HISTORY OF PRESENT ILLNESS / etc.) — closely matches `schema.py`'s fields and is a much more honest generalization check than a random split of the training distribution |

Both were officially released alongside peer-reviewed work (ACI-BENCH:
*Nature Scientific Data*, 2023) and used in the ACL-ClinicalNLP MEDIQA-Chat
2023 / CLEF MEDIQA-SUM 2023 shared tasks — citable, standard benchmarks, not
scraped or ad hoc data. Neither requires PhysioNet-style credentialing, so
both are safe to reference (and MTS-Dialog's actual rows are safe to
redistribute) in a public repo.

`evaluate.py` reports metrics on both: `mts_dialog_test` (held-out split of
the training distribution) and `aci_bench_external` (fully independent —
treat this one as the more meaningful number when you write up results).

MIMIC remains a possible **private, local-only** addition later if credentialed
access comes through — see the section below on why it's excluded from the
public repo itself.

## Quickstart

Two requirements files: `requirements-colab.txt` is what the notebooks actually
install (everything needed for data prep, RAG, DPO training, and eval — nothing
more). `requirements.txt` is the full set, needed only on the machine actually
running `agents.py` (it pulls in SGLang, which the training-side notebooks never
touch). Installing the full file in Colab works too, just noticeably slower for
no benefit, since SGLang's dependency tree is large and unused there.

```bash
pip install -r requirements.txt          # local machine, for agents.py
# or: pip install -r requirements-colab.txt   # Colab, for everything else
python -m spacy download en_core_web_sm   # NER fallback if scispaCy unavailable

# 1. Build data + RAG index (MTS-Dialog for training, ACI-Bench as held-out external eval)
python src/data.py
python src/rag.py --build-index

# 2. Generate preference pairs with the composite reward
python src/preference_pairs.py --n-candidates 4 --n-examples 300

# 3. Supervised fine-tuning baseline (the "supervised baseline" the RLHF stage compares against)
python src/train_sft.py

# 4. DPO fine-tune (LoRA, starts from the SFT checkpoint)
python src/train_dpo.py

# 5. Evaluate base vs. SFT vs. DPO-tuned
python src/evaluate.py

# 6. Merge LoRA into a full model (do this on Colab, then download the result)
python src/merge_lora.py

# 7. (Optional but needed for "correlation with expert judgment") sample outputs for human rating
python src/human_eval.py --mode generate --model-path outputs/dpo_model_merged --n 30
#    ... fill in outputs/human_eval_sheet.csv, then:
python src/human_eval.py --mode score

# 8. Run the multi-agent pipeline end-to-end (on your local PC, needs an SGLang server running)
python -m sglang.launch_server --model-path outputs/dpo_model_merged --port 30000 &
python src/agents.py --dialogue "Doctor: ... Patient: ..."
```

Or open `notebooks/01_setup_data_and_rag.ipynb` in Colab and work through
the six notebooks in order — see `notebooks/README.md` for what each one
produces and roughly how long it takes. They're split up deliberately so a
Colab disconnect partway through doesn't cost you the whole run; each one
mounts Google Drive and picks up wherever the last one left off.

## Results (fill in after your run — do not publish placeholder/aspirational numbers)

| Metric | Base (zero-shot) | SFT baseline | DPO-tuned |
|---|---|---|---|
| ROUGE-L | | | |
| Entity-F1 | | | |
| Malformed-output rate | | | |
| Mean fact-consistency (NLI) | | | |

**Correlation with expert judgment:** r = ___ (Pearson, N = ___ rated examples,
rated by: ___). See `src/human_eval.py`.

Every number above should be copied directly from `outputs/eval_results.json`
and `human_eval.py`'s printed output — not estimated or aspirational. If a
resume bullet cites a specific percentage, it should trace back to a number
in this table from an actual run.

## Honest scope note

This is a one-day, single-GPU build meant to demonstrate the *architecture* and
*methodology* faithfully at small scale (1.5B–3B model, LoRA, ~300–500 training
examples, DPO instead of full PPO). It is not a claim of matching production-scale
numbers — see the README's results table for what your specific run actually produced.

**Two specific claims worth being precise about if you write this up:**

- **"Reward model"** — `reward.py` is a *designed composite reward function*
  (fixed weights over off-the-shelf NER + NLI + a heuristic fluency check), not
  a neural network trained on human preference labels. That's a legitimate and
  common RLHF-adjacent pattern, but it's a different thing than a trained reward
  model in the strict InstructGPT sense — say "designed a reward function," not
  "trained a reward model," if asked directly.
- **"Faster convergence"** — this repo does not include a controlled comparison
  that would support a convergence-speed claim. Measuring that faithfully would
  require running two comparable training loops (e.g., DPO vs. PPO, or reward-
  shaped vs. unshaped) and comparing steps/time to reach a comparable
  quality plateau, which is out of scope for a same-day build. Either drop this
  metric from your writeup, or reframe it honestly as future work.
