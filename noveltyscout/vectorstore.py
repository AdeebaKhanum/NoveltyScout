"""ChromaDB persistence and semantic retrieval for NoveltyScout."""
from __future__ import annotations

from pathlib import Path
import chromadb

from . import config


def _clean_meta(doc: dict) -> dict:
    """Chroma metadata supports scalar values only; normalise missing values."""
    try:
        year = int(doc.get("year") or 0)
    except (TypeError, ValueError):
        year = 0
    return {
        "title": str(doc.get("title") or "Untitled document")[:300],
        "kind": str(doc.get("kind") or "paper").lower(),
        "source": str(doc.get("source") or "unknown"),
        "origin": str(doc.get("origin") or "live"),
        "year": year,
        "url": str(doc.get("url") or ""),
        "source_document_id": str(doc.get("source_document_id") or doc.get("id") or ""),
    }


class PriorArtStore:
    def __init__(self, path=config.CHROMA_DIR, name: str = config.COLLECTION_NAME):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(self.path))
        self.collection = self.client.get_or_create_collection(
            name=name, metadata={"hnsw:space": "cosine"}
        )

    def add_documents(self, docs: list[dict]) -> int:
        """Insert or update valid documents. Existing IDs are updated, not duplicated."""
        if not isinstance(docs, list):
            return 0
        valid = [doc for doc in docs if isinstance(doc, dict) and doc.get("id")
                 and (doc.get("text") or doc.get("abstract") or doc.get("title"))]
        if not valid:
            return 0
        # Chroma requires unique IDs in one upsert request.
        unique: dict[str, dict] = {}
        for doc in valid:
            unique[str(doc["id"])] = doc
        ids = list(unique)
        texts = [str(doc.get("text") or
                     f"{doc.get('title', '')}. {doc.get('abstract', '')}").strip()
                 for doc in unique.values()]
        metas = [_clean_meta(doc) for doc in unique.values()]
        self.collection.upsert(ids=ids, documents=texts, metadatas=metas)
        return len(ids)

    def delete_origin(self, origin: str) -> None:
        """Delete all records belonging to a specific source origin."""
        if origin:
            self.collection.delete(where={"origin": str(origin)})

    def count(self) -> int:
        return int(self.collection.count())

    def count_by_origin(self) -> dict:
        result: dict[str, int] = {}
        # Paginate because a Chroma collection can exceed the default get() page size.
        total = self.count()
        batch_size = 1000
        for offset in range(0, total, batch_size):
            batch = self.collection.get(
                limit=batch_size, offset=offset, include=["metadatas"]
            )
            for metadata in batch.get("metadatas") or []:
                origin = str((metadata or {}).get("origin") or "unknown")
                result[origin] = result.get(origin, 0) + 1
        return result

    def query(self, text: str, k: int = config.TOP_K,
              kind: str | None = None) -> list[dict]:
        """Return best-first semantic matches with cosine similarity in [approx. -1, 1]."""
        query_text = str(text or "").strip()
        try:
            limit = max(1, int(k))
        except (TypeError, ValueError):
            limit = config.TOP_K
        if not query_text or self.count() == 0:
            return []

        kwargs = {"query_texts": [query_text], "n_results": min(limit, self.count())}
        if kind:
            kwargs["where"] = {"kind": str(kind).lower()}
        result = self.collection.query(**kwargs)
        ids = (result.get("ids") or [[]])[0]
        documents = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]

        hits = []
        for index, doc_id in enumerate(ids):
            metadata = metadatas[index] if index < len(metadatas) else {}
            distance = distances[index] if index < len(distances) else 1.0
            hits.append({
                "id": doc_id,
                "text": documents[index] if index < len(documents) else "",
                "similarity": round(1.0 - float(distance), 4),
                **(metadata or {}),
            })
        return hits
