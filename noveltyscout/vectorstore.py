"""Vector database layer (acceptance criterion 7) built on ChromaDB.

How it works (viva summary):
  1. Each document is stored as text = "title. abstract".
  2. ChromaDB converts the text to an embedding (a list of numbers) using its default
     embedding model (all-MiniLM-L6-v2, 384 dimensions, runs locally via ONNX).
  3. Embeddings are indexed with HNSW (approximate nearest-neighbour graph) using COSINE distance.
  4. query() embeds the question the same way and returns the closest documents.
     similarity = 1 - cosine_distance  (1.0 = identical direction, 0 = unrelated).
"""
from __future__ import annotations

import chromadb

from . import config


def _clean_meta(doc: dict) -> dict:
    """Chroma metadata may only hold str/int/float/bool - never None."""
    return {
        "title": str(doc.get("title", ""))[:300],
        "kind": str(doc.get("kind", "paper")),          # "patent" | "paper"
        "source": str(doc.get("source", "")),
        "origin": str(doc.get("origin", "live")),       # "sample" | "live" | "upload"
        "year": int(doc.get("year") or 0),
        "url": str(doc.get("url", "")),
    }


class PriorArtStore:
    def __init__(self, path=config.CHROMA_DIR, name: str = config.COLLECTION_NAME):
        self.client = chromadb.PersistentClient(path=str(path))
        self.collection = self.client.get_or_create_collection(
            name=name, metadata={"hnsw:space": "cosine"}
        )

    # ---------------------------------------------------------------- write
    def add_documents(self, docs: list[dict]) -> int:
        """Upsert documents (same id twice = overwrite, so no duplicates). Returns count."""
        docs = [d for d in docs if d.get("id") and (d.get("text") or d.get("abstract"))]
        if not docs:
            return 0
        ids = [d["id"] for d in docs]
        texts = [d.get("text") or f"{d.get('title', '')}. {d.get('abstract', '')}" for d in docs]
        metas = [_clean_meta(d) for d in docs]
        self.collection.upsert(ids=ids, documents=texts, metadatas=metas)
        return len(ids)

    def delete_origin(self, origin: str) -> None:
        """Remove every document of one origin (e.g. forget previous live results)."""
        self.collection.delete(where={"origin": origin})

    # ----------------------------------------------------------------- read
    def count(self) -> int:
        return self.collection.count()

    def count_by_origin(self) -> dict:
        got = self.collection.get(include=["metadatas"])
        counts: dict = {}
        for meta in got.get("metadatas") or []:
            counts[meta["origin"]] = counts.get(meta["origin"], 0) + 1
        return counts

    def query(self, text: str, k: int = config.TOP_K, kind: str | None = None) -> list[dict]:
        """Similarity search. Returns dicts sorted best-first, each with a 'similarity' field."""
        total = self.count()
        if total == 0 or not text.strip():
            return []
        kwargs = {"query_texts": [text], "n_results": min(k, total)}
        if kind:
            kwargs["where"] = {"kind": kind}
        res = self.collection.query(**kwargs)
        out = []
        for i, doc_id in enumerate(res["ids"][0]):
            meta = res["metadatas"][0][i]
            out.append({
                "id": doc_id,
                "text": res["documents"][0][i],
                "similarity": round(1.0 - float(res["distances"][0][i]), 4),
                **meta,
            })
        return out
