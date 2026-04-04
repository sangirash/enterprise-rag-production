"""
Versioned prompt templates for the RAG pipeline.

Keeping prompts as versioned constants makes it straightforward to A/B test
prompt changes and audit which prompt version produced a given response.
"""
from __future__ import annotations

RAG_SYSTEM_PROMPT = """\
You are a precise, factual assistant. Answer the user's question using ONLY the context provided.
If the context does not contain sufficient information to answer the question, say so explicitly.
Do not speculate, hallucinate, or draw on knowledge outside of the provided context.
When citing information, reference the numbered source tags (e.g. [1], [2]) from the context.\
"""

FAITHFULNESS_CHECK_PROMPT = """\
You are evaluating whether an answer is grounded in the provided context.
For each factual claim in the answer, determine whether it is directly supported
by information in the context. Return only a numeric score between 0.0 and 1.0,
where 1.0 means every claim is fully supported and 0.0 means no claims are supported.\
"""


def build_rag_prompt(
    query: str,
    context_chunks: list[str],
    metadata: list[dict] | None = None,
) -> str:
    """
    Assemble the user-turn prompt from a query and a list of context chunks.

    Each chunk is labeled with a numbered citation tag and, optionally, a
    source identifier drawn from the metadata list.
    """
    parts: list[str] = []
    for i, chunk in enumerate(context_chunks, start=1):
        source_label = ""
        if metadata and i - 1 < len(metadata):
            src = metadata[i - 1].get("source_id", "")
            if src:
                source_label = f" [Source: {src}]"
        parts.append(f"[{i}]{source_label}\n{chunk}")

    context_block = "\n\n---\n\n".join(parts)
    return f"Context:\n\n{context_block}\n\nQuestion: {query}\n\nAnswer:"
