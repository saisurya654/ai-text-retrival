from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from app.models.schemas import DocumentChunk, RetrievalHit
from app.utils.logging import get_logger
from retrieval.embeddings import EmbeddingService

logger = get_logger(__name__)


class VectorStore:
    def __init__(self, storage_path: Path) -> None:
        self.storage_path = storage_path
        self.embedding_service = EmbeddingService()
        self.chunks: list[DocumentChunk] = []
        self.embeddings: np.ndarray | None = None
        self._index = None
        self._faiss = None
        self._load()

    def add(self, chunks: list[DocumentChunk]) -> None:
        if not chunks:
            return
        existing_ids = {chunk.chunk_id for chunk in self.chunks}
        chunks = [chunk for chunk in chunks if chunk.chunk_id not in existing_ids]
        if not chunks:
            return
        new_embeddings = self.embedding_service.encode(chunk.text for chunk in chunks)
        self.chunks.extend(chunks)
        self.embeddings = new_embeddings if self.embeddings is None else np.vstack([self.embeddings, new_embeddings])
        self._rebuild_index()
        self._persist()

    def search(self, query: str, top_k: int = 5) -> list[RetrievalHit]:
        if not self.chunks or self.embeddings is None:
            return []

        query_vector = self.embedding_service.encode([query])[0]
        scores, indices = self._query_index(query_vector, top_k)
        hits: list[RetrievalHit] = []
        for score, idx in zip(scores, indices):
            if idx < 0 or idx >= len(self.chunks):
                continue
            chunk = self.chunks[idx]
            hits.append(
                RetrievalHit(
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    page_number=chunk.page_number,
                    score=float(score),
                    text=chunk.text,
                    metadata=chunk.metadata,
                )
            )
        return hits

    def _query_index(self, query_vector: np.ndarray, top_k: int) -> tuple[list[float], list[int]]:
        if self._index is not None and self._faiss is not None:
            distances, indices = self._index.search(np.asarray([query_vector], dtype="float32"), top_k)
            return distances[0].tolist(), indices[0].tolist()

        similarities = self.embeddings @ query_vector
        ranked = np.argsort(-similarities)[:top_k]
        return similarities[ranked].tolist(), ranked.tolist()

    def _rebuild_index(self) -> None:
        try:
            import faiss  # type: ignore

            self._faiss = faiss
            dims = int(self.embeddings.shape[1])
            index = faiss.IndexFlatIP(dims)
            index.add(np.asarray(self.embeddings, dtype="float32"))
            self._index = index
        except Exception as exc:  # pragma: no cover - native dependency
            logger.warning("FAISS unavailable, using numpy cosine search: %s", exc)
            self._index = None
            self._faiss = None

    def _persist(self) -> None:
        payload = {
            "chunks": [chunk.model_dump() for chunk in self.chunks],
            "embeddings": self.embeddings.tolist() if self.embeddings is not None else [],
        }
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self.storage_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _load(self) -> None:
        if not self.storage_path.exists():
            return
        payload = json.loads(self.storage_path.read_text(encoding="utf-8"))
        self.chunks = [DocumentChunk(**item) for item in payload.get("chunks", [])]
        embeddings = payload.get("embeddings", [])
        if embeddings:
            self.embeddings = np.asarray(embeddings, dtype="float32")
            self._rebuild_index()
