"""Where prior-art evidence comes from.

Source                 Type     Needs key?  Notes
---------------------  -------  ----------  -------------------------------------------------
Sample corpus (JSON)   patent*  no          Illustrative demo records (NOT real patents)
arXiv API              paper    no          Free Atom/XML API
OpenAlex API           paper    no          Free JSON API, ~250M scholarly works
User upload / paste    both     no          Student pastes abstracts/claims copied from
                                            Google Patents, Espacenet or Indian Patent Search

Why no live patent API?  PatentsView's public API was shut down in March 2026 and the USPTO
Open Data Portal now needs a signed-in account + API key. Google Patents has no official API.
So patents enter the system through the upload tab, and every live patent source can be added
later as one more function that returns the same document dict.
"""
from __future__ import annotations

import hashlib
import json
import re
import xml.etree.ElementTree as ET
from urllib.parse import quote_plus

import requests

from . import config

STOPWORDS = set(
    "a an the and or of for to in on with by from that this is are be as at it its into using "
    "use used which can has have not no than then their there these those over under via based "
    "method system device apparatus comprising wherein".split()
)


class SourceError(Exception):
    """Raised when a live source cannot be reached or parsed."""


# ------------------------------------------------------------------ helpers
def keywords(query: str, limit: int = 6) -> list[str]:
    words = re.findall(r"[a-zA-Z][a-zA-Z\-]{2,}", query.lower())
    seen, out = set(), []
    for w in words:
        if w not in STOPWORDS and w not in seen:
            seen.add(w)
            out.append(w)
    return out[:limit]


def make_doc(doc_id, title, abstract, kind, source, origin, year=0, url="") -> dict:
    title = re.sub(r"\s+", " ", title or "").strip()
    abstract = re.sub(r"\s+", " ", abstract or "").strip()
    return {
        "id": doc_id, "title": title, "abstract": abstract,
        "text": f"{title}. {abstract}".strip(),
        "kind": kind, "source": source, "origin": origin,
        "year": int(year or 0), "url": url,
    }


def google_patents_link(query: str) -> str:
    """Link for MANUAL patent searching (we do not scrape Google Patents)."""
    return "https://patents.google.com/?q=" + quote_plus(query)


# -------------------------------------------------------------- sample data
def load_sample_corpus() -> list[dict]:
    with open(config.SAMPLE_CORPUS, encoding="utf-8") as fh:
        raw = json.load(fh)
    return [
        make_doc(r["id"], r["title"], r["abstract"], r["kind"], "sample-corpus", "sample",
                 r.get("year", 0), "")
        for r in raw
    ]


# ------------------------------------------------------------------- arXiv
def search_arxiv(query: str, max_results: int = config.RESULTS_PER_SOURCE) -> list[dict]:
    terms = keywords(query, 4)
    if not terms:
        return []

    def run(operator: str) -> list[dict]:
        q = f" {operator} ".join(f"all:{t}" for t in terms)
        try:
            resp = requests.get(
                "https://export.arxiv.org/api/query",
                params={"search_query": q, "start": 0, "max_results": max_results,
                        "sortBy": "relevance"},
                timeout=(3, config.HTTP_TIMEOUT),
            )
            resp.raise_for_status()
            return parse_arxiv_xml(resp.text)
        except (requests.RequestException, ET.ParseError) as exc:
            raise SourceError(f"arXiv unavailable: {exc}") from exc

    docs = run("AND")
    return docs or run("OR")        # AND is precise; fall back to OR if nothing matched


def parse_arxiv_xml(xml_text: str) -> list[dict]:
    ns = {"a": "http://www.w3.org/2005/Atom"}
    root = ET.fromstring(xml_text)
    docs = []
    for entry in root.findall("a:entry", ns):
        url = (entry.findtext("a:id", default="", namespaces=ns) or "").strip()
        arxiv_id = url.rsplit("/", 1)[-1]
        year = (entry.findtext("a:published", default="", namespaces=ns) or "")[:4]
        docs.append(make_doc(
            f"arxiv:{arxiv_id}",
            entry.findtext("a:title", default="", namespaces=ns),
            entry.findtext("a:summary", default="", namespaces=ns),
            "paper", "arXiv", "live", year or 0, url,
        ))
    return docs


# ----------------------------------------------------------------- OpenAlex
def search_openalex(query: str, max_results: int = config.RESULTS_PER_SOURCE) -> list[dict]:
    params = {
        "search": " ".join(keywords(query, 8)) or query,
        "per-page": max_results,
        "select": "id,title,publication_year,abstract_inverted_index,doi",
    }
    if config.OPENALEX_EMAIL:
        params["mailto"] = config.OPENALEX_EMAIL
    try:
        resp = requests.get("https://api.openalex.org/works", params=params,
                            timeout=(3, config.HTTP_TIMEOUT))
        resp.raise_for_status()
        return parse_openalex_json(resp.json())
    except (requests.RequestException, ValueError) as exc:
        raise SourceError(f"OpenAlex unavailable: {exc}") from exc


def rebuild_abstract(inverted_index: dict | None) -> str:
    """OpenAlex stores abstracts as {word: [positions]}; put the words back in order."""
    if not inverted_index:
        return ""
    slots: dict[int, str] = {}
    for word, positions in inverted_index.items():
        for pos in positions:
            slots[pos] = word
    return " ".join(slots[i] for i in sorted(slots))


def parse_openalex_json(payload: dict) -> list[dict]:
    docs = []
    for work in payload.get("results", []):
        abstract = rebuild_abstract(work.get("abstract_inverted_index"))
        if not abstract:
            continue                         # a title alone is weak evidence - skip
        short_id = (work.get("id") or "").rsplit("/", 1)[-1]
        docs.append(make_doc(
            f"openalex:{short_id}", work.get("title") or "", abstract,
            "paper", "OpenAlex", "live", work.get("publication_year") or 0,
            work.get("doi") or work.get("id") or "",
        ))
    return docs


# ------------------------------------------------------- user-provided text
def chunk_text(text: str, size: int = config.CHUNK_WORDS,
               overlap: int = config.CHUNK_OVERLAP) -> list[str]:
    """Split long text into overlapping word windows so each chunk embeds well."""
    words = text.split()
    if len(words) <= size:
        return [" ".join(words)] if words else []
    step = max(1, size - overlap)
    return [" ".join(words[i:i + size]) for i in range(0, len(words), step)
            if words[i:i + size]]


def docs_from_text(title: str, text: str, kind: str = "patent", url: str = "") -> list[dict]:
    """Turn pasted/uploaded text into chunked documents (origin='upload')."""
    title = title.strip() or "Uploaded document"
    base = hashlib.md5((title + text[:200]).encode("utf-8")).hexdigest()[:8]
    chunks = chunk_text(text)
    return [
        make_doc(f"UPLOAD-{base}-{i + 1}", f"{title} (part {i + 1})" if len(chunks) > 1 else title,
                 chunk, kind, "user-upload", "upload", 0, url)
        for i, chunk in enumerate(chunks)
    ]


def text_from_pdf(file_obj) -> str:
    from pypdf import PdfReader

    reader = PdfReader(file_obj)
    return "\n".join((page.extract_text() or "") for page in reader.pages)
