"""
This is the reward function -- not a trained model, just three signals combined
with fixed weights. Used twice: once offline to label which candidate summary is
"better" when building DPO pairs, and again at inference time inside the verifier
agent to decide if a draft needs another pass.

reward = w1 * entity_F1 + w2 * fact_consistency + w3 * discourse_flow

If scispaCy's clinical NER model isn't installed we just fall back to plain
spaCy. Entity typing is worse but at least nothing crashes.
"""
import re

from config import CFG

_nlp = None
_nli = None


def _get_nlp():
    global _nlp
    if _nlp is not None:
        return _nlp
    import spacy
    for model_name in [CFG.ner_model_scispacy, CFG.ner_model_fallback]:
        try:
            _nlp = spacy.load(model_name)
            print(f"Loaded NER model: {model_name}")
            return _nlp
        except OSError:
            continue
    raise RuntimeError(
        "No NER model available. Run: python -m spacy download en_core_web_sm"
    )


def _get_nli():
    global _nli
    if _nli is not None:
        return _nli
    from sentence_transformers import CrossEncoder
    _nli = CrossEncoder(CFG.nli_model_name)
    return _nli


def extract_entities(text: str) -> set:
    doc = _get_nlp()(text)
    return {ent.text.lower().strip() for ent in doc.ents if ent.text.strip()}


def entity_f1(source: str, summary: str) -> float:
    src_ents = extract_entities(source)
    sum_ents = extract_entities(summary)
    if not sum_ents:
        return 0.0
    if not src_ents:
        return 0.0
    tp = len(src_ents & sum_ents)
    precision = tp / len(sum_ents) if sum_ents else 0.0
    recall = tp / len(src_ents) if src_ents else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def fact_consistency(source: str, summary: str) -> float:
    """1 - P(contradiction) from an NLI model, treating source as premise."""
    model = _get_nli()
    # cross-encoder/nli-deberta-v3-base label order: contradiction, entailment, neutral
    scores = model.predict([(source[:2000], summary[:1000])])  # truncate for context length
    import numpy as np
    probs = np.exp(scores) / np.exp(scores).sum(axis=-1, keepdims=True)
    contradiction_prob = float(probs[0][0])
    return 1.0 - contradiction_prob


def discourse_flow(summary: str) -> float:
    """
    Rough fluency check, no extra model needed. Just penalizes text that's too
    short, too repetitive, or reads like a run-on with no real sentence breaks.
    Honestly the weakest of the three signals -- good enough to catch garbage
    output, not sophisticated enough to judge actual writing quality.
    """
    words = summary.split()
    if len(words) < 5:
        return 0.1

    unique_ratio = len(set(w.lower() for w in words)) / len(words)
    sentences = re.split(r"[.!?]+", summary)
    sentences = [s for s in sentences if s.strip()]
    avg_sent_len = len(words) / max(len(sentences), 1)

    length_penalty = 1.0 if 4 <= avg_sent_len <= 35 else 0.5
    score = 0.6 * unique_ratio + 0.4 * length_penalty
    return max(0.0, min(1.0, score))


def composite_reward(source: str, summary: str) -> dict:
    ef1 = entity_f1(source, summary)
    fc = fact_consistency(source, summary)
    df = discourse_flow(summary)
    w = CFG.reward_weights
    total = w["entity_f1"] * ef1 + w["fact_consistency"] * fc + w["discourse_flow"] * df
    return {
        "entity_f1": ef1,
        "fact_consistency": fc,
        "discourse_flow": df,
        "total": total,
    }


if __name__ == "__main__":
    src = ("Doctor: What brings you in today? Patient: I've had chest pain for "
           "two days and shortness of breath. Doctor: Any history of heart disease? "
           "Patient: My father had a heart attack at 55.")
    good = "Chief complaint: chest pain and shortness of breath for two days. Family history of MI."
    bad = "Chief complaint: headache and fever for a week. No relevant history."
    print("Good summary reward:", composite_reward(src, good))
    print("Bad summary reward:", composite_reward(src, bad))
