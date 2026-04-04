"""
RAG evaluation metrics.

Faithfulness    -- fraction of answer claims grounded in retrieved context (LLM judge).
Context Precision -- fraction of retrieved chunks relevant to the query (embedding similarity).
Answer Relevance  -- semantic similarity between the query and the generated answer.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from openai import OpenAI
from sentence_transformers import SentenceTransformer

from ..config import settings

_client = OpenAI(api_key=settings.openai_api_key)
_embedder = SentenceTransformer("all-MiniLM-L6-v2")


@dataclass
class EvalResult:
    query: str
    answer: str
    faithfulness: float
    context_precision: float
    answer_relevance: float

    @property
    def overall(self) -> float:
        return (self.faithfulness + self.context_precision + self.answer_relevance) / 3


def evaluate_faithfulness(answer: str, context_chunks: list[str]) -> float:
    """
    Ask GPT-4o-mini to judge whether every factual claim in the answer
    is directly supported by the provided context chunks.

    Returns a float in [0.0, 1.0].
    """
    context = "\n\n".join(context_chunks)
    prompt = (
        f"Context:\n{context}\n\n"
        f"Answer:\n{answer}\n\n"
        "For each factual claim in the answer, determine if it is directly supported "
        "by the context above. Return a score between 0.0 and 1.0 where 1.0 means "
        "all claims are fully supported. Return only the numeric score."
    )
    response = _client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    try:
        return float(response.choices[0].message.content.strip())  # type: ignore[arg-type]
    except (ValueError, AttributeError):
        return 0.0


def evaluate_context_precision(query: str, context_chunks: list[str]) -> float:
    """
    Compute the fraction of retrieved chunks whose embedding is sufficiently
    similar to the query embedding (cosine similarity > threshold).
    """
    if not context_chunks:
        return 0.0
    query_emb = _embedder.encode(query, convert_to_numpy=True)
    chunk_embs = _embedder.encode(context_chunks, convert_to_numpy=True)
    sims = np.dot(chunk_embs, query_emb) / (
        np.linalg.norm(chunk_embs, axis=1) * np.linalg.norm(query_emb) + 1e-8
    )
    threshold = 0.4
    return float(np.mean(sims > threshold))


def evaluate_answer_relevance(query: str, answer: str) -> float:
    """Cosine similarity between query and answer embeddings."""
    embs = _embedder.encode([query, answer], convert_to_numpy=True)
    sim = float(
        np.dot(embs[0], embs[1])
        / (np.linalg.norm(embs[0]) * np.linalg.norm(embs[1]) + 1e-8)
    )
    return max(0.0, min(1.0, sim))
