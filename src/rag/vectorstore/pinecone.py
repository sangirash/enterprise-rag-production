from __future__ import annotations

from typing import Any

from .base import BaseVectorStore
from ..config import settings


class PineconeStore(BaseVectorStore):
    """
    Pinecone serverless vector store adapter.

    Content and source_id are stored in Pinecone metadata alongside the vector.
    Batches upserts to stay within Pinecone's per-request limits.
    """

    _UPSERT_BATCH = 100

    def __init__(self) -> None:
        from pinecone import Pinecone  # type: ignore[import]

        pc = Pinecone(api_key=settings.pinecone_api_key)
        self.index = pc.Index(settings.pinecone_index)

    def upsert(self, records: list[dict[str, Any]]) -> None:
        vectors = [
            {
                "id": r["id"],
                "values": r["embedding"],
                "metadata": {
                    **r.get("metadata", {}),
                    "content": r["content"],
                    "source_id": r.get("source_id", ""),
                },
            }
            for r in records
        ]
        for i in range(0, len(vectors), self._UPSERT_BATCH):
            self.index.upsert(vectors=vectors[i : i + self._UPSERT_BATCH])

    def similarity_search(
        self,
        embedding: list[float],
        top_k: int = 10,
    ) -> list[tuple[dict[str, Any], float]]:
        result = self.index.query(vector=embedding, top_k=top_k, include_metadata=True)
        return [
            (
                {
                    "id": match.id,
                    "content": match.metadata.get("content", ""),
                    "metadata": {
                        k: v for k, v in match.metadata.items() if k not in ("content", "source_id")
                    },
                    "source_id": match.metadata.get("source_id", ""),
                },
                float(match.score),
            )
            for match in result.matches
        ]

    def delete(self, ids: list[str]) -> None:
        self.index.delete(ids=ids)
