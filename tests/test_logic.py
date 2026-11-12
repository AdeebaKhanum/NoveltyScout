"""Offline tests for NoveltyScout's deterministic logic and source adapters.

Run with: python -m pytest -q
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from noveltyscout import scoring  # noqa: E402
from noveltyscout.memory import SessionMemory  # noqa: E402
from noveltyscout.sources import (  # noqa: E402
    chunk_text,
    docs_from_text,
    keywords,
    parse_arxiv_xml,
    parse_openalex_json,
    rebuild_abstract,
)


def test_overlap_index():
    comps = [
        {"verdict": "disclosed"},
        {"verdict": "partial"},
        {"verdict": "not_found"},
        {"verdict": "not_found"},
    ]
    assert scoring.overlap_index(comps) == 37.5
    assert scoring.overlap_index([]) == 0.0
    assert scoring.overlap_index([None, {"verdict": "disclosed"}]) == 100.0


def test_overlap_band_is_neutral():
    assert "limited matching evidence" in scoring.overlap_band(0)
    assert "some matching evidence" in scoring.overlap_band(40)
    assert "substantial matching evidence" in scoring.overlap_band(80)
    assert "promising" not in scoring.overlap_band(0).lower()


def test_single_reference_coverage_flag():
    comps = [
        {"verdict": "disclosed", "evidence_ids": ["A", "B"]},
        {"verdict": "disclosed", "evidence_ids": ["A"]},
    ]
    flags = scoring.risk_flags(comps)
    assert flags["anticipation_risk"]
    assert flags["closest_reference"] == "A"
    assert not flags["combination_review"]


def test_multi_reference_signal_is_not_called_legal_obviousness():
    comps = [
        {"verdict": "disclosed", "evidence_ids": ["A"]},
        {"verdict": "disclosed", "evidence_ids": ["B"]},
    ]
    flags = scoring.risk_flags(comps)
    assert flags["combination_review"]
    assert "not a finding of obviousness" in flags["note"]
    assert not flags["anticipation_risk"]


def test_partial_feature_does_not_create_combination_flag():
    comps = [
        {"verdict": "disclosed", "evidence_ids": ["A"]},
        {"verdict": "partial", "evidence_ids": ["B"]},
    ]
    assert not scoring.risk_flags(comps)["combination_review"]


def test_validation_drops_invented_ids_and_invalid_verdicts():
    raw = [
        {"verdict": "disclosed", "evidence_ids": ["FAKE-9"], "reasoning": "x"},
        {"verdict": "partial", "evidence_ids": ["S-1"], "reasoning": "y"},
        {"verdict": "banana", "evidence_ids": ["S-1"]},
    ]
    rows = scoring.validate_comparisons(
        raw, ["f1", "f2", "f3"], {0: {"S-1"}, 1: {"S-1"}, 2: {"S-1"}}
    )
    assert [row["verdict"] for row in rows] == ["not_found", "partial", "not_found"]
    assert rows[0]["evidence_ids"] == []


def test_validation_handles_string_id_and_duplicate_ids():
    rows = scoring.validate_comparisons(
        [{"verdict": "disclosed", "evidence_ids": "S-1"}],
        ["feature"], {0: {"S-1"}},
    )
    assert rows[0]["evidence_ids"] == ["S-1"]


def test_chunking():
    text = " ".join(f"w{i}" for i in range(500))
    chunks = chunk_text(text, size=100, overlap=20)
    assert len(chunks) >= 5 and all(len(chunk.split()) <= 100 for chunk in chunks)
    assert chunk_text("short text") == ["short text"]
    assert chunk_text("  ") == []


def test_chunking_rejects_invalid_overlap():
    try:
        chunk_text("some words", size=5, overlap=5)
    except ValueError:
        pass
    else:
        raise AssertionError("Expected invalid overlap to raise ValueError")


def test_openalex_abstract_rebuild():
    inverted = {"hello": [0], "world": [1], "again": [2]}
    assert rebuild_abstract(inverted) == "hello world again"
    docs = parse_openalex_json({"results": [{
        "id": "https://openalex.org/W1",
        "title": "T",
        "publication_year": 2020,
        "abstract_inverted_index": inverted,
        "doi": "10.1234/example",
    }]})
    assert docs[0]["id"] == "openalex:W1"
    assert docs[0]["url"] == "https://doi.org/10.1234/example"


def test_arxiv_parse():
    xml = """<feed xmlns="http://www.w3.org/2005/Atom"><entry>
    <id>http://arxiv.org/abs/2101.00001v1</id><published>2021-01-01T00:00:00Z</published>
    <title> A  title </title><summary>Some abstract.</summary></entry></feed>"""
    docs = parse_arxiv_xml(xml)
    assert docs[0]["id"] == "arxiv:2101.00001v1"
    assert docs[0]["title"] == "A title"
    assert docs[0]["year"] == 2021


def test_keywords():
    assert keywords("A method of using the tilt sensor for tablets") == ["tilt", "sensor", "tablets"]
    assert keywords("tilt sensor", limit=0) == []


def test_uploaded_document_ids_are_deterministic_and_distinct():
    first = docs_from_text("Document", "An abstract with useful content.")
    second = docs_from_text("Document", "A different abstract with useful content.")
    assert first[0]["id"] != second[0]["id"]
    assert first[0]["origin"] == "upload"
    assert docs_from_text("Empty", "") == []


def test_uploaded_chunks_are_grouped_as_one_reference():
    long_text = " ".join(f"term{index}" for index in range(450))
    chunks = docs_from_text("One patent", long_text)
    assert len(chunks) > 1
    assert len({chunk["source_document_id"] for chunk in chunks}) == 1
    evidence = {chunk["id"]: chunk for chunk in chunks}
    comparisons = [
        {"verdict": "disclosed", "evidence_ids": [chunks[0]["id"]]},
        {"verdict": "disclosed", "evidence_ids": [chunks[-1]["id"]]},
    ]
    flags = scoring.risk_flags(comparisons, evidence)
    assert flags["anticipation_risk"]


def test_session_history_is_bounded():
    memory = SessionMemory()
    for index in range(60):
        memory.add_turn("user", f"message {index}")
    assert len(memory.history) == 40
    assert memory.history[-1][1] == "message 59"


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items())
             if name.startswith("test_") and callable(value)]
    for test in tests:
        test()
        print("PASS", test.__name__)
    print(f"{len(tests)} tests passed")
