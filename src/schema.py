"""
Defines the structured clinical note schema shared by:
  - SGLang constrained decoding (json schema below)
  - the malformed-output check in evaluate.py
"""
from typing import Optional

from pydantic import BaseModel, Field


class ClinicalNote(BaseModel):
    chief_complaint: str = Field(..., description="One-line reason for the visit")
    history_of_present_illness: str = Field(..., description="Narrative of the current issue")
    assessment: str = Field(..., description="Clinician's assessment / impression")
    plan: str = Field(..., description="Next steps: meds, follow-up, referrals")
    medications_mentioned: Optional[str] = Field(
        default="", description="Comma-separated medications mentioned, or empty string"
    )


# JSON schema used to constrain SGLang generation (gen(..., json_schema=...))
CLINICAL_NOTE_JSON_SCHEMA = ClinicalNote.model_json_schema()


def is_well_formed(raw_json_str: str) -> bool:
    """Returns True if the model output parses into a valid ClinicalNote."""
    try:
        ClinicalNote.model_validate_json(raw_json_str)
        return True
    except Exception:  # noqa: BLE001
        return False


def note_to_text(note: ClinicalNote) -> str:
    """Flatten a structured note into plain text for scoring (ROUGE, NER, NLI)."""
    parts = [
        note.chief_complaint,
        note.history_of_present_illness,
        note.assessment,
        note.plan,
    ]
    if note.medications_mentioned:
        parts.append(note.medications_mentioned)
    return " ".join(p for p in parts if p)
