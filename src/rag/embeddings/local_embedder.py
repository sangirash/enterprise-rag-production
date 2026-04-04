from __future__ import annotations

from sentence_transformers import SentenceTransformer

from .base import BaseEmbedder


class LocalEmbedder(BaseEmbedder):
    """
    Local sentence-transformers embedding model.
    Fully offline -- no API calls, no rate limits.
    Default model: all-MiniLM-L6-v2 (384 dimensions, fast inference).
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.model = SentenceTransformer(model_name)

    def embed(self, text: str) -> list[float]:
        return self.model.encode(text, convert_to_numpy=True).tolist()  # type: ignore[union-attr]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return [emb.tolist() for emb in embeddings]  # type: ignore[union-attr]
