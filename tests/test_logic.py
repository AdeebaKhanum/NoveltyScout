"""Offline unit tests for the pure-Python parts (no API key, no internet, no LangChain needed
for most of them).  Run:  python tests/test_logic.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from noveltyscout import scoring                      # noqa: E402
from noveltyscout.sources import (chunk_text, parse_arxiv_xml, parse_openalex_json,  # noqa: E402
                                 rebuild_abstract, keywords)


def test_overlap_index():
    comps = [{"verdict": "disclosed"}, {"verdict": "partial"}, {"verdict": "not_found"}, {"verdict": "not_found"}]
    assert scoring.overlap_index(comps) == 37.5
    assert scoring.overlap_index([]) == 0.0


def test_anticipation_flag():
    comps = [{"verdict": "disclosed", "evidence_ids": ["A", "B"]},
             {"verdict": "disclosed", "evidence_ids": ["A"]}]
    flags = scoring.risk_flags(comps)
    assert flags["anticipation_risk"] and flags["closest_reference"] == "A"


def test_obviousness_flag():
    comps = [{"verdict": "disclosed", "evidence_ids": ["A"]},
             {"verdict": "disclosed", "evidence_ids": ["B"]}]
    flags = scoring.risk_flags(comps)
    assert flags["obviousness_risk"] and not flags["anticipation_risk"]


def test_validation_drops_invented_ids():
    raw = [{"verdict": "disclosed", "evidence_ids": ["FAKE-9"], "reasoning": "x"},
           {"verdict": "partial", "evidence_ids": ["S-1"], "reasoning": "y"},
           {"verdict": "banana", "evidence_ids": ["S-1"]}]
    rows = scoring.validate_comparisons(raw, ["f1", "f2", "f3"], {0: {"S-1"}, 1: {"S-1"}, 2: {"S-1"}})
    assert [r["verdict"] for r in rows] == ["not_found", "partial", "not_found"]
    assert rows[0]["evidence_ids"] == []


def test_chunking():
    text = " ".join(f"w{i}" for i in range(500))
    chunks = chunk_text(text, size=100, overlap=20)
    assert len(chunks) >= 5 and all(len(c.split()) <= 100 for c in chunks)
    assert chunk_text("short text") == ["short text"]


def test_openalex_abstract_rebuild():
    inv = {"hello": [0], "world": [1], "again": [2]}
    assert rebuild_abstract(inv) == "hello world again"
    docs = parse_openalex_json({"results": [{"id": "https://openalex.org/W1", "title": "T",
                                             "publication_year": 2020, "abstract_inverted_index": inv}]})
    assert docs[0]["id"] == "openalex:W1"


def test_arxiv_parse():
    xml = """<feed xmlns="http://www.w3.org/2005/Atom"><entry>
    <id>http://arxiv.org/abs/2101.00001v1</id><published>2021-01-01T00:00:00Z</published>
    <title> A  title </title><summary>Some abstract.</summary></entry></feed>"""
    docs = parse_arxiv_xml(xml)
    assert docs[0]["id"] == "arxiv:2101.00001v1" and docs[0]["title"] == "A title" and docs[0]["year"] == 2021


def test_keywords():
    assert keywords("A method of using the tilt sensor for tablets") == ["tilt", "sensor", "tablets"]


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print("PASS", t.__name__)
    print(f"{len(tests)} tests passed")
