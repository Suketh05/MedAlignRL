"""
Loads the MTS-Dialog dataset (open, no credentialing needed) and produces
train/val/test JSONL splits with a consistent schema:
    {"id": ..., "dialogue": ..., "summary": ...}

If the primary HF dataset id is unavailable, falls back to alternate mirrors.
"""
import json
import os
import random

from datasets import load_dataset

from config import CFG
from schema import ClinicalNote

CANDIDATE_DATASET_IDS = [
    "har1/MTS_Dialogue-Clinical_Note",
    "blabbertok/mts-dialogue",
    "AGBonnet/augmented-clinical-notes",  # larger, alt fallback (dialogue+note style)
]

# Field-name candidates seen across MTS-Dialog mirrors.
DIALOGUE_FIELDS = ["dialogue", "conversation", "text", "src"]
SUMMARY_FIELDS = ["section_text", "summary", "note", "tgt"]
HEADER_FIELDS = ["section_header", "header", "section"]


def _first_present(row, candidates):
    for c in candidates:
        if c in row and row[c]:
            return row[c]
    return None


def load_raw():
    last_err = None
    for ds_id in CANDIDATE_DATASET_IDS:
        try:
            print(f"Trying dataset: {ds_id}")
            ds = load_dataset(ds_id)
            split = "train" if "train" in ds else list(ds.keys())[0]
            return ds[split], ds_id
        except Exception as e:  # noqa: BLE001
            print(f"  failed: {e}")
            last_err = e
    raise RuntimeError(
        "Could not load any MTS-Dialog mirror. Last error: "
        f"{last_err}. Check dataset name on huggingface.co/datasets."
    )


def build_splits():
    random.seed(CFG.seed)
    raw, ds_id = load_raw()

    examples = []
    for i, row in enumerate(raw):
        dialogue = _first_present(row, DIALOGUE_FIELDS)
        summary = _first_present(row, SUMMARY_FIELDS)
        if not dialogue or not summary:
            continue
        header = _first_present(row, HEADER_FIELDS)
        examples.append({
            "id": f"{i}",
            "dialogue": dialogue.strip(),
            "summary": summary.strip(),
            "section_header": header.strip() if header else "",
        })

    print(f"Loaded {len(examples)} usable examples from {ds_id}")
    random.shuffle(examples)

    n = len(examples)
    n_train = int(n * CFG.train_split)
    n_val = int(n * CFG.val_split)

    splits = {
        "train": examples[:n_train],
        "val": examples[n_train:n_train + n_val],
        "test": examples[n_train + n_val:],
    }

    for name, subset in splits.items():
        path = os.path.join(CFG.data_dir, f"{name}.jsonl")
        with open(path, "w") as f:
            for ex in subset:
                f.write(json.dumps(ex) + "\n")
        print(f"Wrote {len(subset)} examples to {path}")

    return splits


def load_split(name):
    path = os.path.join(CFG.data_dir, f"{name}.jsonl")
    if not os.path.exists(path):
        build_splits()
    with open(path) as f:
        return [json.loads(line) for line in f]


# ---------------------------------------------------------------------------
# MTS-Dialog rows are single sections (one section_header + section_text),
# not full notes -- so converting one to a ClinicalNote necessarily leaves
# every field but the mapped one empty. Used by train_sft.py so its
# fine-tuning target is schema-shaped JSON instead of raw section text,
# matching what evaluate.py/human_eval.py actually prompt for.
# ---------------------------------------------------------------------------

# Exact section_header codes, as used by the primary MTS-Dialog mirror.
_EXACT_HEADER_FIELD_MAP = {
    "CC": "chief_complaint",
    "GENHX": "history_of_present_illness",
    "ASSESSMENT": "assessment",
    "DIAGNOSIS": "assessment",
    "PLAN": "plan",
    "DISPOSITION": "plan",
    "ALLERGY": "medications_mentioned",
    "ALLERGIES": "medications_mentioned",
    "MEDICATIONS": "medications_mentioned",
}

# Substring fallback for mirrors that spell headers out in full instead of
# using the short codes above.
_SUBSTRING_HEADER_FIELD_MAP = {
    "CHIEF COMPLAINT": "chief_complaint",
    "HISTORY OF PRESENT ILLNESS": "history_of_present_illness",
    "IMPRESSION": "assessment",
    "ASSESSMENT": "assessment",
    "DIAGNOSIS": "assessment",
    "DISPOSITION": "plan",
    "PLAN": "plan",
    "ALLERGY": "medications_mentioned",
    "MEDICATION": "medications_mentioned",
}


def _map_header_to_field(section_header):
    header = (section_header or "").strip().upper()
    if not header:
        return None
    if header in _EXACT_HEADER_FIELD_MAP:
        return _EXACT_HEADER_FIELD_MAP[header]
    for keyword, field in _SUBSTRING_HEADER_FIELD_MAP.items():
        if keyword in header:
            return field
    return None


def row_to_clinical_note_json(section_header: str, section_text: str) -> str:
    """Converts one MTS-Dialog (section_header, section_text) row into a
    ClinicalNote-shaped JSON string."""
    fields = {
        "chief_complaint": "",
        "history_of_present_illness": "",
        "assessment": "",
        "plan": "",
        "medications_mentioned": "",
    }
    field = _map_header_to_field(section_header)
    if field:
        fields[field] = section_text.strip()
    return ClinicalNote(**fields).model_dump_json()


# ---------------------------------------------------------------------------
# ACI-Bench: kept completely separate from training. Full visit dialogues with
# full structured notes, which lines up well with schema.py's fields. Used
# only as a held-out sanity check -- if the model does well here, it's
# actually generalizing, not just memorizing the MTS-Dialog distribution.
# https://huggingface.co/datasets/mkieffer/ACI-Bench
# ---------------------------------------------------------------------------
ACI_BENCH_DATASET_ID = "mkieffer/ACI-Bench"
ACI_BENCH_SUBSET = "aci"  # natural doctor-patient dialogue subset (vs. virtassist/virtscribe)
ACI_BENCH_TEST_SPLITS = ["test1", "test2", "test3"]  # 22+22+22 = 66 examples


def build_aci_bench_eval():
    """Builds data/aci_bench_eval.jsonl -- an external, held-out eval set.
    Never used for training or for building the RAG index."""
    all_rows = []
    for split in ACI_BENCH_TEST_SPLITS:
        ds = load_dataset(ACI_BENCH_DATASET_ID, name=ACI_BENCH_SUBSET, split=split)
        for row in ds:
            all_rows.append({
                "id": f"{split}_{row['encounter_id']}",
                "dialogue": row["dialogue"].strip(),
                "summary": row["note"].strip(),
            })

    path = os.path.join(CFG.data_dir, "aci_bench_eval.jsonl")
    with open(path, "w") as f:
        for ex in all_rows:
            f.write(json.dumps(ex) + "\n")
    print(f"Wrote {len(all_rows)} ACI-Bench held-out eval examples to {path}")
    return all_rows


def load_aci_bench_eval():
    path = os.path.join(CFG.data_dir, "aci_bench_eval.jsonl")
    if not os.path.exists(path):
        build_aci_bench_eval()
    with open(path) as f:
        return [json.loads(line) for line in f]


if __name__ == "__main__":
    build_splits()
    build_aci_bench_eval()
