from __future__ import annotations

from dataclasses import replace

from sentence_transformers import CrossEncoder

from .retriever import RetrievedChunk


class CrossEncoderReranker:
    """
    Cross-encoder reranker for precision-optimised top-N selection.

    Unlike bi-encoders, a cross-encoder reads the full (query, document) pair
    jointly, enabling richer interaction but requiring O(n) model forward passes.
    Apply only to the short candidate list produced by the initial retrieval step.

    Default model: cross-encoder/ms-marco-MiniLM-L-6-v2
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        top_n: int = 4,
    ) -> None:
        self.model = CrossEncoder(model_name)
        self.top_n = top_n

    def rerank(self, query: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        if not chunks:
            return []
        pairs = [(query, c.content) for c in chunks]
        scores = self.model.predict(pairs)
        ranked = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)
        return [replace(c, score=float(s)) for c, s in ranked[: self.top_n]]
