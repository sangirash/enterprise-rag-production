from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class Chunk:
    content: str
    metadata: dict[str, Any]
    chunk_index: int
    source_id: str
    token_count: int


class BaseChunker(ABC):
    @abstractmethod
    def chunk(self, text: str, metadata: dict[str, Any], source_id: str) -> list[Chunk]:
        pass
