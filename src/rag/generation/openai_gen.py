from __future__ import annotations

import structlog
from openai import OpenAI

from .base import BaseGenerator
from .prompt_templates import RAG_SYSTEM_PROMPT
from ..config import settings

logger = structlog.get_logger()


class OpenAIGenerator(BaseGenerator):
    """
    OpenAI GPT-4o generation backend.

    Temperature is set to 0 for deterministic, factual responses.
    System prompt enforces grounding in retrieved context.
    """

    def __init__(self) -> None:
        self.client = OpenAI(api_key=settings.openai_api_key)
        self.model = settings.openai_model

    def generate(self, query: str, context: str) -> str:
        user_message = f"Context:\n\n{context}\n\nQuestion: {query}"
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": RAG_SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=0,
            max_tokens=1024,
        )
        answer = response.choices[0].message.content or ""
        logger.debug(
            "generation_complete",
            model=self.model,
            prompt_tokens=response.usage.prompt_tokens if response.usage else None,
            completion_tokens=response.usage.completion_tokens if response.usage else None,
        )
        return answer
