from abc import ABC, abstractmethod
from typing import Any


class BaseVectorStore(ABC):
    """Abstract interface for vector store backends."""

    @abstractmethod
    def upsert(self, records: list[dict[str, Any]]) -> None:
        """Insert or update records. Each record must have: id, content, embedding, metadata, source_id."""
        pass

    @abstractmethod
    def similarity_search(
        self,
        embedding: list[float],
        top_k: int = 10,
    ) -> list[tuple[dict[str, Any], float]]:
        """Return top_k (chunk_dict, score) pairs ordered by descending relevance."""
        pass

    @abstractmethod
    def delete(self, ids: list[str]) -> None:
        """Delete records by ID."""
        pass
