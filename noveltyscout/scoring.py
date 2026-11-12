"""Deterministic scoring (plain Python, NO LLM).

Why not let the LLM compute the score?  LLMs are unreliable at arithmetic and give different
numbers on each run.  The LLM only supplies per-feature VERDICTS; the numbers and risk flags
below are computed by code, so they are reproducible and explainable in a viva.

Patent logic mirrored here
  * Novelty (anticipation): ONE earlier document discloses ALL elements of the invention.
  * Inventive step (obviousness): the elements are all known but spread across several documents,
    so combining them may be obvious to a skilled person.
"""
from __future__ import annotations

VERDICT_WEIGHT = {"disclosed": 1.0, "partial": 0.5, "not_found": 0.0}
VALID_VERDICTS = set(VERDICT_WEIGHT)


def overlap_index(comparisons: list[dict]) -> float:
    """Average verdict weight x 100  ->  0 (nothing known) .. 100 (everything known)."""
    if not comparisons:
        return 0.0
    total = sum(VERDICT_WEIGHT.get(c.get("verdict"), 0.0) for c in comparisons)
    return round(100.0 * total / len(comparisons), 1)


def overlap_band(score: float) -> str:
    if score < 25:
        return "LOW overlap - idea looks promising"
    if score < 55:
        return "MODERATE overlap - some elements are known"
    return "HIGH overlap - most elements already known"


def reference_coverage(comparisons: list[dict]) -> dict[str, list[int]]:
    """For each reference ID, which feature numbers does it DISCLOSE (verdict == 'disclosed')?"""
    cover: dict[str, list[int]] = {}
    for idx, comp in enumerate(comparisons):
        if comp.get("verdict") != "disclosed":
            continue
        for ref_id in comp.get("evidence_ids", []):
            cover.setdefault(ref_id, []).append(idx)
    return cover


def risk_flags(comparisons: list[dict]) -> dict:
    """Return anticipation / obviousness flags from the comparison table."""
    n = len(comparisons)
    cover = reference_coverage(comparisons)
    best_ref, best_count = (None, 0)
    for ref_id, feats in cover.items():
        if len(feats) > best_count:
            best_ref, best_count = ref_id, len(feats)

    anticipation = n > 0 and best_count == n                      # one reference has everything
    all_disclosed = n > 0 and all(c.get("verdict") == "disclosed" for c in comparisons)
    obviousness = all_disclosed and not anticipation              # known, but spread over several refs
    if anticipation:
        note = f"Reference {best_ref} appears to disclose ALL features (possible lack of novelty)."
    elif obviousness:
        note = "Every feature is known, but across different references (possible lack of inventive step)."
    elif best_ref:
        note = (f"Closest single reference: {best_ref} covers {best_count} of {n} features; "
                "no single reference covers all of them.")
    else:
        note = "No reference clearly discloses any feature."
    return {
        "anticipation_risk": anticipation,
        "obviousness_risk": obviousness,
        "closest_reference": best_ref,
        "closest_coverage": f"{best_count}/{n}",
        "note": note,
    }


def validate_comparisons(raw: list, features: list[str], allowed_ids: dict[int, set]) -> list[dict]:
    """Clean the LLM's verdict list.

    * forces exactly one row per feature, in order
    * unknown verdicts become 'not_found'
    * evidence IDs the model invented (not in that feature's retrieved set) are dropped
    * 'disclosed'/'partial' with no valid evidence is downgraded to 'not_found'
    """
    rows = []
    raw = raw if isinstance(raw, list) else []
    for i, feature in enumerate(features):
        item = raw[i] if i < len(raw) and isinstance(raw[i], dict) else {}
        verdict = str(item.get("verdict", "not_found")).strip().lower()
        if verdict not in VALID_VERDICTS:
            verdict = "not_found"
        ids = [e for e in item.get("evidence_ids", []) if e in allowed_ids.get(i, set())]
        if verdict != "not_found" and not ids:
            verdict = "not_found"
        if verdict == "not_found":
            ids = []
        rows.append({
            "feature": feature,
            "verdict": verdict,
            "evidence_ids": ids,
            "reasoning": str(item.get("reasoning", "")).strip(),
        })
    return rows
