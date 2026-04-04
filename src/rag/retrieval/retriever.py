from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from rank_bm25 import BM25Okapi


@dataclass
class RetrievedChunk:
    content: str
    metadata: dict[str, Any]
    source_id: str
    score: float
    retrieval_method: str


class HybridRetriever:
    """
    Combines dense vector search with BM25 sparse retrieval using score fusion.

    Final score = alpha * dense_score_normalized + (1 - alpha) * bm25_score_normalized

    alpha=1.0 means pure dense retrieval; alpha=0.0 means pure BM25.
    Default alpha=0.7 favors semantic relevance while retaining keyword recall.
    """

    def __init__(self, vector_store: Any, alpha: float = 0.7, top_k: int = 10) -> None:
        self.vector_store = vector_store
        self.alpha = alpha
        self.top_k = top_k
        self._bm25: BM25Okapi | None = None
        self._corpus: list[dict[str, Any]] = []

    def index_corpus(self, chunks: list[dict[str, Any]]) -> None:
        """Build an in-memory BM25 index from the provided chunk list."""
        self._corpus = chunks
        tokenized = [c["content"].lower().split() for c in chunks]
        self._bm25 = BM25Okapi(tokenized)

    def retrieve(self, query: str, query_embedding: list[float]) -> list[RetrievedChunk]:
        dense_results = self.vector_store.similarity_search(
            query_embedding, top_k=self.top_k * 2
        )

        if self._bm25 and self._corpus:
            bm25_scores = self._bm25.get_scores(query.lower().split())
            top_idx = np.argsort(bm25_scores)[::-1][: self.top_k * 2]
            bm25_results: list[tuple[dict[str, Any], float]] = [
                (self._corpus[i], float(bm25_scores[i])) for i in top_idx
            ]
        else:
            bm25_results = []

        return self._fuse(dense_results, bm25_results)

    def _fuse(
        self,
        dense: list[tuple[dict[str, Any], float]],
        sparse: list[tuple[dict[str, Any], float]],
    ) -> list[RetrievedChunk]:
        scores: dict[str, float] = {}
        chunks: dict[str, dict[str, Any]] = {}

        max_dense = max((s for _, s in dense), default=1.0) or 1.0
        for chunk, score in dense:
            cid = chunk["id"]
            scores[cid] = scores.get(cid, 0.0) + self.alpha * (score / max_dense)
            chunks[cid] = chunk

        max_sparse = max((s for _, s in sparse), default=1.0) or 1.0
        for chunk, score in sparse:
            cid = chunk["id"]
            scores[cid] = scores.get(cid, 0.0) + (1 - self.alpha) * (score / max_sparse)
            chunks[cid] = chunk

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[: self.top_k]
        return [
            RetrievedChunk(
                content=chunks[cid]["content"],
                metadata=chunks[cid].get("metadata", {}),
                source_id=chunks[cid].get("source_id", ""),
                score=score,
                retrieval_method="hybrid",
            )
            for cid, score in ranked
        ]
