"""
Ingestion pipeline orchestrator.

Wires together: document loading -> preprocessing -> chunking -> embedding -> vector store upsert.
"""
from __future__ import annotations

import asyncio
import hashlib
from typing import Any

import structlog

from .loader import load_from_bytes
from .preprocessor import preprocess
from ..chunking.fixed import FixedSizeChunker
from ..chunking.recursive import RecursiveChunker
from ..chunking.semantic import SemanticChunker
from ..embeddings.openai_embedder import OpenAIEmbedder
from ..vectorstore.pgvector import PgVectorStore
from ..vectorstore.pinecone import PineconeStore
from ..vectorstore.weaviate import WeaviateStore
from ..config import settings

logger = structlog.get_logger()


class IngestionPipeline:
    """Stateless orchestrator for the document ingestion path."""

    def __init__(self) -> None:
        chunker_map = {
            "fixed": FixedSizeChunker(settings.chunk_size, settings.chunk_overlap),
            "recursive": RecursiveChunker(settings.chunk_size, settings.chunk_overlap),
            "semantic": SemanticChunker(max_chunk_tokens=settings.chunk_size),
        }
        self.chunker = chunker_map[settings.chunk_strategy]
        self.embedder = OpenAIEmbedder()

        store_map: dict[str, Any] = {
            "pgvector": PgVectorStore,
            "pinecone": PineconeStore,
            "weaviate": WeaviateStore,
        }
        self.vector_store = store_map[settings.vector_store]()

    async def ingest(
        self,
        content: bytes,
        filename: str,
        source_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Ingest a document from raw bytes.

        Returns a dict with 'chunks_created' indicating how many chunks were stored.
        """
        doc = load_from_bytes(content, filename, source_id)
        processed = preprocess(doc)
        if processed is None:
            logger.warning("ingestion_skipped", source_id=source_id, reason="empty_or_duplicate")
            return {"chunks_created": 0}

        extra = metadata or {}
        chunks = self.chunker.chunk(
            processed.content,
            {**processed.metadata, **extra},
            source_id,
        )
        logger.info("document_chunked", source_id=source_id, chunk_count=len(chunks))

        texts = [c.content for c in chunks]
        embeddings = await asyncio.to_thread(self.embedder.embed_batch, texts)

        records = [
            {
                "id": hashlib.sha256(f"{source_id}:{c.chunk_index}".encode()).hexdigest()[:16],
                "content": c.content,
                "embedding": emb,
                "metadata": c.metadata,
                "source_id": c.source_id,
            }
            for c, emb in zip(chunks, embeddings)
        ]
        await asyncio.to_thread(self.vector_store.upsert, records)
        logger.info("document_ingested", source_id=source_id, chunks_stored=len(records))
        return {"chunks_created": len(records)}
