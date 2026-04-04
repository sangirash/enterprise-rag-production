from typing import Any
from sentence_transformers import SentenceTransformer
import numpy as np
from .base import BaseChunker, Chunk
import tiktoken


class SemanticChunker(BaseChunker):
    """Semantic chunker: splits at points of maximum embedding cosine dissimilarity."""

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        breakpoint_threshold: float = 0.3,
        max_chunk_tokens: int = 512,
    ):
        self.model = SentenceTransformer(model_name)
        self.threshold = breakpoint_threshold
        self.max_tokens = max_chunk_tokens
        self.encoder = tiktoken.encoding_for_model("gpt-4o")

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

    def chunk(self, text: str, metadata: dict[str, Any], source_id: str) -> list[Chunk]:
        sentences = [s.strip() for s in text.replace("\n", " ").split(". ") if s.strip()]
        if not sentences:
            return []

        embeddings = self.model.encode(sentences, convert_to_numpy=True)
        breakpoints = []
        for i in range(1, len(sentences)):
            sim = self._cosine_similarity(embeddings[i - 1], embeddings[i])
            if sim < (1.0 - self.threshold):
                breakpoints.append(i)

        chunks = []
        prev = 0
        for bp in breakpoints + [len(sentences)]:
            segment = ". ".join(sentences[prev:bp]) + "."
            tokens = self.encoder.encode(segment)
            chunks.append(Chunk(
                content=segment,
                metadata={**metadata, "chunk_strategy": "semantic"},
                chunk_index=len(chunks),
                source_id=source_id,
                token_count=len(tokens),
            ))
            prev = bp
        return chunks
