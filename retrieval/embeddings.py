from __future__ import annotations

import hashlib
from typing import Iterable

import numpy as np

from app.utils.logging import get_logger

logger = get_logger(__name__)


class EmbeddingService:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.model_name = model_name
        self._model = None

    def _ensure_model(self):
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore

            self._model = SentenceTransformer(self.model_name)
        except Exception as exc:  # pragma: no cover - model install dependent
            logger.warning("SentenceTransformer unavailable, using hash embeddings: %s", exc)
            self._model = False
        return self._model

    def encode(self, texts: Iterable[str]) -> np.ndarray:
        text_list = list(texts)
        model = self._ensure_model()
        if model is False:
            return np.vstack([self._hash_vector(text) for text in text_list])
        vectors = model.encode(text_list, normalize_embeddings=True)
        return np.asarray(vectors, dtype="float32")

    def _hash_vector(self, text: str, dims: int = 384) -> np.ndarray:
        seed = hashlib.sha256(text.encode("utf-8")).digest()
        values = np.frombuffer((seed * ((dims // len(seed)) + 1))[:dims], dtype=np.uint8)
        vector = values.astype("float32")
        norm = np.linalg.norm(vector) or 1.0
        return vector / norm
