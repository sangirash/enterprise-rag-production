from __future__ import annotations

import hashlib
import time
from typing import Any

import structlog
from openai import OpenAI, RateLimitError

from .base import BaseEmbedder
from ..config import settings

logger = structlog.get_logger()

# Module-level embedding cache keyed by content hash.
_CACHE: dict[str, list[float]] = {}


class OpenAIEmbedder(BaseEmbedder):
    """
    OpenAI embedding client.

    Features:
    - Batches requests (max 100 texts per API call).
    - Exponential backoff on RateLimitError (up to MAX_RETRIES attempts).
    - Content-hash cache to avoid redundant API calls within a process lifetime.
    """

    BATCH_SIZE = 100
    MAX_RETRIES = 5
    BASE_DELAY = 1.0

    def __init__(self) -> None:
        self.client = OpenAI(api_key=settings.openai_api_key)
        self.model = settings.embedding_model

    def _cache_key(self, text: str) -> str:
        return hashlib.sha256(f"{self.model}:{text}".encode()).hexdigest()

    def embed(self, text: str) -> list[float]:
        key = self._cache_key(text)
        if key in _CACHE:
            return _CACHE[key]
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        results: list[list[float] | None] = [None] * len(texts)
        uncached_indices: list[int] = []
        uncached_texts: list[str] = []

        for i, text in enumerate(texts):
            key = self._cache_key(text)
            if key in _CACHE:
                results[i] = _CACHE[key]
            else:
                uncached_indices.append(i)
                uncached_texts.append(text)

        if uncached_texts:
            fresh = self._embed_with_retry(uncached_texts)
            for idx, emb, text in zip(uncached_indices, fresh, uncached_texts):
                _CACHE[self._cache_key(text)] = emb
                results[idx] = emb

        return [r for r in results if r is not None]

    def _embed_with_retry(self, texts: list[str]) -> list[list[float]]:
        all_embeddings: list[list[float]] = []
        for batch_start in range(0, len(texts), self.BATCH_SIZE):
            batch = texts[batch_start : batch_start + self.BATCH_SIZE]
            delay = self.BASE_DELAY
            for attempt in range(self.MAX_RETRIES):
                try:
                    response = self.client.embeddings.create(model=self.model, input=batch)
                    sorted_data = sorted(response.data, key=lambda x: x.index)
                    all_embeddings.extend(item.embedding for item in sorted_data)
                    break
                except RateLimitError:
                    if attempt == self.MAX_RETRIES - 1:
                        raise
                    logger.warning("openai_rate_limit", attempt=attempt, backoff_seconds=delay)
                    time.sleep(delay)
                    delay *= 2
        return all_embeddings
