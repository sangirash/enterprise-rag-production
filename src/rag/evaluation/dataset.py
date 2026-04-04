"""
Synthetic evaluation dataset generation.

Uses GPT-4o-mini to generate factual question-answer pairs from provided context chunks.
Useful for bootstrapping an eval dataset when no labeled QA pairs exist.
"""
from __future__ import annotations

import json

from openai import OpenAI

from ..config import settings
from .harness import EvalCase

_client = OpenAI(api_key=settings.openai_api_key)


def generate_eval_dataset(
    context_chunks: list[str],
    n_questions: int = 10,
) -> list[EvalCase]:
    """
    Generate a synthetic eval dataset from context chunks.

    Args:
        context_chunks: Source passages used to generate questions.
        n_questions: Target number of QA pairs to generate.

    Returns:
        List of EvalCase instances with query and expected_answer populated.
    """
    context = "\n\n".join(context_chunks[:20])
    prompt = (
        f"Given the following context, generate {n_questions} diverse factual questions "
        "that can be answered directly from the context. "
        'Return a JSON object with a "questions" array. '
        'Each element must have "query" and "expected_answer" string fields.\n\n'
        f"Context:\n{context}"
    )
    response = _client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        response_format={"type": "json_object"},
    )
    raw = response.choices[0].message.content or "{}"
    data = json.loads(raw)
    items = data.get("questions", data.get("items", []))
    return [
        EvalCase(
            query=item["query"],
            expected_answer=item.get("expected_answer"),
        )
        for item in items
        if "query" in item
    ]
