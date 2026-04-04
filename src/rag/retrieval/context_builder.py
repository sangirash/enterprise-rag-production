from __future__ import annotations

from typing import Any

from .retriever import RetrievedChunk


class ContextBuilder:
    """
    Assembles retrieved chunks into a formatted context window with source attribution.

    Each chunk is prefixed with a numbered citation marker so that downstream
    prompts can reference specific sources and the API can return structured
    provenance information.
    """

    def build(self, query: str, chunks: list[RetrievedChunk]) -> dict[str, Any]:
        context_parts: list[str] = []
        sources: list[dict[str, Any]] = []

        for i, chunk in enumerate(chunks, start=1):
            context_parts.append(f"[{i}] {chunk.content}")
            sources.append(
                {
                    "index": i,
                    "source_id": chunk.source_id,
                    "score": round(chunk.score, 4),
                    "retrieval_method": chunk.retrieval_method,
                    "metadata": chunk.metadata,
                }
            )

        return {
            "context_text": "\n\n".join(context_parts),
            "sources": sources,
            "query": query,
        }
