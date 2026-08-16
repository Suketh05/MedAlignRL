"""
The multi-agent clinical summarization pipeline.

    Retriever -> Drafter (SGLang, JSON-schema constrained) -> Verifier (reward.py)
              -> Refiner (only if verifier score is low) -> final structured note

Run an SGLang server first:
    python -m sglang.launch_server --model-path <model_path_or_hf_id> --port 30000

Then:
    python src/agents.py --dialogue "Doctor: ... Patient: ..."
"""
import argparse
import json

import sglang as sgl

from config import CFG
from rag import Retriever
from reward import composite_reward
from schema import CLINICAL_NOTE_JSON_SCHEMA, is_well_formed, ClinicalNote

DRAFT_INSTRUCTIONS = """You are a clinical scribe. Read the dialogue and produce a \
structured clinical note. Only include facts explicitly stated in the dialogue — \
never invent symptoms, medications, or history not mentioned.

Reference example (style only, do not copy facts):
Dialogue: {ex_dialogue}
Note: {ex_summary}

Dialogue to summarize:
{dialogue}
"""

REFINE_INSTRUCTIONS = """The following structured clinical note may contain facts \
not supported by the dialogue, or may be missing key facts. Revise it so every \
statement is directly supported by the dialogue below. Keep it structured and concise.

Dialogue:
{dialogue}

Draft note (JSON):
{draft_json}

Issues detected: entity overlap with source was low and/or possible contradiction \
with the source was detected. Produce a corrected note.
"""


@sgl.function
def draft_agent(s, dialogue: str, ex_dialogue: str, ex_summary: str):
    s += DRAFT_INSTRUCTIONS.format(
        dialogue=dialogue, ex_dialogue=ex_dialogue[:600], ex_summary=ex_summary[:400]
    )
    s += sgl.gen(
        "note_json",
        max_tokens=400,
        json_schema=json.dumps(CLINICAL_NOTE_JSON_SCHEMA),
    )


@sgl.function
def refine_agent(s, dialogue: str, draft_json: str):
    s += REFINE_INSTRUCTIONS.format(dialogue=dialogue, draft_json=draft_json)
    s += sgl.gen(
        "note_json",
        max_tokens=400,
        json_schema=json.dumps(CLINICAL_NOTE_JSON_SCHEMA),
    )


class ClinicalSummarizationPipeline:
    """Retriever -> Drafter -> Verifier -> (optional) Refiner."""

    def __init__(self, sglang_endpoint: str = "http://localhost:30000"):
        sgl.set_default_backend(sgl.RuntimeEndpoint(sglang_endpoint))
        self.retriever = Retriever()
        self.retriever.load()

    def run(self, dialogue: str) -> dict:
        # 1. Retriever agent
        exemplars = self.retriever.retrieve(dialogue, k=1)
        ex = exemplars[0] if exemplars else {"dialogue": "", "summary": ""}

        # 2. Drafter agent (SGLang, schema-constrained)
        state = draft_agent.run(
            dialogue=dialogue, ex_dialogue=ex["dialogue"], ex_summary=ex["summary"]
        )
        draft_json = state["note_json"]
        malformed = not is_well_formed(draft_json)

        # 3. Verifier agent
        from schema import note_to_text
        note_text = draft_json
        verifier_score = {"total": 0.0}
        if not malformed:
            note = ClinicalNote.model_validate_json(draft_json)
            note_text = note_to_text(note)
            verifier_score = composite_reward(dialogue, note_text)

        result = {
            "dialogue": dialogue,
            "retrieved_exemplar_similarity": ex.get("similarity"),
            "draft": draft_json,
            "malformed": malformed,
            "verifier_score": verifier_score,
            "refined": False,
        }

        # 4. Refiner agent (only if needed)
        if malformed or verifier_score["total"] < CFG.verifier_score_threshold:
            refine_state = refine_agent.run(dialogue=dialogue, draft_json=draft_json)
            refined_json = refine_state["note_json"]
            refined_malformed = not is_well_formed(refined_json)
            refined_score = {"total": 0.0}
            if not refined_malformed:
                refined_note = ClinicalNote.model_validate_json(refined_json)
                refined_score = composite_reward(dialogue, note_to_text(refined_note))

            result.update({
                "refined": True,
                "final": refined_json,
                "final_malformed": refined_malformed,
                "final_score": refined_score,
            })
        else:
            result.update({
                "final": draft_json,
                "final_malformed": malformed,
                "final_score": verifier_score,
            })

        return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dialogue", type=str, required=True)
    parser.add_argument("--endpoint", type=str, default="http://localhost:30000")
    args = parser.parse_args()

    pipeline = ClinicalSummarizationPipeline(sglang_endpoint=args.endpoint)
    output = pipeline.run(args.dialogue)
    print(json.dumps(output, indent=2))
