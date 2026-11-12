"""Deterministic, explainable screening metrics (not a legal patentability test).

The language model supplies feature-level labels. This module validates those labels,
removes unsupported reference IDs, and calculates a reproducible evidence-overlap
indicator. It deliberately does not decide legal novelty or inventive step.
"""
from __future__ import annotations

VERDICT_WEIGHT = {"disclosed": 1.0, "partial": 0.5, "not_found": 0.0}
VALID_VERDICTS = frozenset(VERDICT_WEIGHT)


def overlap_index(comparisons: list[dict]) -> float:
    """Return the mean feature verdict weight as a 0–100 screening indicator."""
    valid_rows = [row for row in comparisons if isinstance(row, dict)]
    if not valid_rows:
        return 0.0
    total = sum(VERDICT_WEIGHT.get(str(row.get("verdict", "")).lower(), 0.0)
                for row in valid_rows)
    return round(100.0 * total / len(valid_rows), 1)


def overlap_band(score: float) -> str:
    """Describe matching-evidence volume without implying a patentability outcome."""
    try:
        value = float(score)
    except (TypeError, ValueError):
        value = 0.0
    if value < 25:
        return "LOW reference overlap - limited matching evidence"
    if value < 55:
        return "MODERATE reference overlap - some matching evidence"
    return "HIGH reference overlap - substantial matching evidence"


def reference_coverage(comparisons: list[dict], evidence_by_id: dict | None = None) -> dict[str, list[int]]:
    """Map each source document to the feature indices it was judged to disclose.

    Uploaded long documents are split into Chroma chunks. Their chunk IDs differ, but
    their ``source_document_id`` metadata links them back to one source document.
    """
    evidence_by_id = evidence_by_id if isinstance(evidence_by_id, dict) else {}
    coverage: dict[str, list[int]] = {}
    for index, comp in enumerate(comparisons):
        if not isinstance(comp, dict) or comp.get("verdict") != "disclosed":
            continue
        ids = comp.get("evidence_ids", [])
        if not isinstance(ids, (list, tuple, set)):
            continue
        for ref_id in ids:
            if not isinstance(ref_id, str) or not ref_id.strip():
                continue
            metadata = evidence_by_id.get(ref_id.strip(), {})
            source_id = metadata.get("source_document_id") if isinstance(metadata, dict) else None
            group_id = str(source_id or ref_id).strip()
            if group_id:
                coverage.setdefault(group_id, []).append(index)
    return coverage


def risk_flags(comparisons: list[dict], evidence_by_id: dict | None = None) -> dict:
    """Summarise single-reference coverage and multi-reference review signals.

    A single reference associated with every disclosed feature is a useful flag for
    human review. Multiple references collectively covering the features is NOT, by
    itself, proof of obviousness; it is labelled only as a combination-review signal.
    """
    rows = [row for row in comparisons if isinstance(row, dict)]
    n = len(rows)
    coverage = reference_coverage(rows, evidence_by_id)
    best_group, best_count = None, 0
    for group_id, feature_indices in coverage.items():
        if len(feature_indices) > best_count:
            best_group, best_count = group_id, len(feature_indices)
    best_ref = best_group
    # Prefer a real retrieved chunk ID in the UI/report when a grouped upload ID
    # is not itself a citeable evidence ID.
    if best_group and isinstance(evidence_by_id, dict):
        for evidence_id, metadata in evidence_by_id.items():
            if isinstance(metadata, dict) and metadata.get("source_document_id") == best_group:
                best_ref = evidence_id
                break

    all_disclosed = n > 0 and all(row.get("verdict") == "disclosed" for row in rows)
    single_reference_coverage = n > 0 and best_count == n
    combination_review = all_disclosed and not single_reference_coverage

    if single_reference_coverage:
        note = (f"Reference {best_ref} was cited as disclosing every checked feature. "
                "Review the original document and claim scope manually.")
    elif combination_review:
        note = ("Every checked feature was matched to evidence, but the matches are spread "
                "across multiple references. This is a signal for further human review, "
                "not a finding of obviousness.")
    elif best_ref:
        note = (f"Closest single reference: {best_ref} was cited for {best_count} of {n} "
                "features. No single reference covers every checked feature.")
    else:
        note = "No retrieved reference was judged to clearly disclose a checked feature."

    # Keep obviousness_risk as a backwards-compatible alias for older UI integrations.
    # New code should use combination_review, which has a more accurate meaning.
    return {
        "anticipation_risk": single_reference_coverage,
        "obviousness_risk": combination_review,
        "combination_review": combination_review,
        "closest_reference": best_ref,
        "closest_coverage": f"{best_count}/{n}",
        "note": note,
    }


def validate_comparisons(raw: list, features: list[str],
                         allowed_ids: dict[int, set]) -> list[dict]:
    """Return exactly one safe, evidence-grounded comparison row per feature."""
    rows = []
    raw_rows = raw if isinstance(raw, list) else []
    for index, feature in enumerate(features):
        item = raw_rows[index] if index < len(raw_rows) and isinstance(raw_rows[index], dict) else {}
        verdict = str(item.get("verdict", "not_found")).strip().lower()
        if verdict not in VALID_VERDICTS:
            verdict = "not_found"

        supplied_ids = item.get("evidence_ids", [])
        if isinstance(supplied_ids, str):
            supplied_ids = [supplied_ids]
        if not isinstance(supplied_ids, (list, tuple, set)):
            supplied_ids = []
        permitted = allowed_ids.get(index, set())
        if not isinstance(permitted, (set, list, tuple)):
            permitted = set()
        permitted = {value for value in permitted if isinstance(value, str)}
        evidence_ids = []
        for ref_id in supplied_ids:
            if isinstance(ref_id, str) and ref_id in permitted and ref_id not in evidence_ids:
                evidence_ids.append(ref_id)

        if verdict != "not_found" and not evidence_ids:
            verdict = "not_found"
        if verdict == "not_found":
            evidence_ids = []

        rows.append({
            "feature": str(feature),
            "verdict": verdict,
            "evidence_ids": evidence_ids,
            "reasoning": str(item.get("reasoning", "")).strip()[:1000],
        })
    return rows
