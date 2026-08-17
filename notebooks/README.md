# Colab notebooks

Run these one at a time, in order, each as its own Colab session. They're
split up on purpose -- Colab sessions time out or disconnect, and losing
three hours of preference-pair generation because the eval step crashed at
the end is not a fun way to spend a day.

All six mount Google Drive and clone/pull the repo into
`/content/drive/MyDrive/MedAlignRL`, so anything a notebook produces
(datasets, indices, checkpoints) is still there the next time you open a
different one, even in a fresh runtime.

| # | Notebook | Produces | Rough time (A100) |
|---|---|---|---|
| 1 | `01_setup_data_and_rag.ipynb` | `data/*.jsonl`, `data/rag_index/` | ~10 min |
| 2 | `02_generate_preference_pairs.ipynb` | `data/preference_pairs.jsonl` | ~45-60 min |
| 3 | `03_train_sft_baseline.ipynb` | `outputs/sft_model/` | ~5-10 min |
| 4 | `04_train_dpo.ipynb` | `outputs/dpo_model/` | ~5-10 min |
| 5 | `05_evaluate.ipynb` | `outputs/eval_results.json` | ~30-40 min |
| 6 | `06_merge_export_and_human_eval.ipynb` | `outputs/dpo_model_merged/` (pushed to HF or downloaded), `outputs/human_eval_sheet.csv`, results pushed to GitHub | ~15 min + however long rating takes |

Notebook 6 merges weights, (optionally) exports the model, and runs human
eval, in that order in a single notebook -- human eval needs the merged
model that the same notebook's first step produces, so splitting them
across two separate Colab sessions just meant always having to remember to
run one before the other.

Before running notebook 1: push your code to GitHub (a private repo is
fine at this stage -- nothing about training requires it to be public yet).
Then, **in notebook 1**, follow the token setup note (uses Colab's Secrets
manager, not a token pasted into the notebook itself -- pasting one directly
would leak it into git history the moment the notebook gets committed) and
set `GITHUB_USERNAME` / `GITHUB_REPO` in the setup cell. If your repo is
public, you can skip the token entirely -- the same cell falls back to a
plain clone.

If you're re-running any of these later with a different model or dataset
config, edit `src/config.py` in your GitHub repo and push -- the next setup
cell's `git fetch` + hard reset picks it up automatically.
