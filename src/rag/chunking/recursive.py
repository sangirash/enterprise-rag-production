from typing import Any
from langchain.text_splitter import RecursiveCharacterTextSplitter
import tiktoken
from .base import BaseChunker, Chunk


class RecursiveChunker(BaseChunker):
    """Recursive character text splitter -- respects paragraph and sentence boundaries."""

    def __init__(self, chunk_size: int = 512, overlap: int = 64):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size * 4,  # approximate char count
            chunk_overlap=overlap * 4,
            separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
        )
        self.encoder = tiktoken.encoding_for_model("gpt-4o")

    def chunk(self, text: str, metadata: dict[str, Any], source_id: str) -> list[Chunk]:
        splits = self.splitter.split_text(text)
        return [
            Chunk(
                content=split,
                metadata={**metadata, "chunk_strategy": "recursive"},
                chunk_index=i,
                source_id=source_id,
                token_count=len(self.encoder.encode(split)),
            )
            for i, split in enumerate(splits)
        ]
