import tiktoken
from typing import Any
from .base import BaseChunker, Chunk


class FixedSizeChunker(BaseChunker):
    """Fixed token-count chunker with configurable overlap."""

    def __init__(self, chunk_size: int = 512, overlap: int = 64, model: str = "gpt-4o"):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.encoder = tiktoken.encoding_for_model(model)

    def chunk(self, text: str, metadata: dict[str, Any], source_id: str) -> list[Chunk]:
        tokens = self.encoder.encode(text)
        chunks = []
        start = 0
        index = 0
        while start < len(tokens):
            end = min(start + self.chunk_size, len(tokens))
            chunk_tokens = tokens[start:end]
            chunk_text = self.encoder.decode(chunk_tokens)
            chunks.append(Chunk(
                content=chunk_text,
                metadata={**metadata, "chunk_strategy": "fixed"},
                chunk_index=index,
                source_id=source_id,
                token_count=len(chunk_tokens),
            ))
            start += self.chunk_size - self.overlap
            index += 1
        return chunks
