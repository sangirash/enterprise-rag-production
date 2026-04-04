"""
RAGPipeline -- main end-to-end orchestrator.

Wires together:
    ingestion (text) -> chunking -> embedding -> vector store upsert
    query -> embedding -> retrieval (hybrid) -> reranking -> context assembly -> generation
"""
from __future__ import annotations

import asyncio
import hashlib
from typing import Any

import structlog

from .chunking.fixed import FixedSizeChunker
from .chunking.recursive import RecursiveChunker
from .chunking.semantic import SemanticChunker
from .config import settings
from .embeddings.openai_embedder import OpenAIEmbedder
from .generation.openai_gen import OpenAIGenerator
from .retrieval.context_builder import ContextBuilder
from .retrieval.reranker import CrossEncoderReranker
from .retrieval.retriever import HybridRetriever
from .vectorstore.pgvector import PgVectorStore
from .vectorstore.pinecone import PineconeStore
from .vectorstore.weaviate import WeaviateStore

logger = structlog.get_logger()


class RAGPipeline:
    """
    Top-level RAG pipeline.

    Instantiating this class is cheap -- heavy models (cross-encoder, sentence-transformers)
    are loaded lazily by their respective classes on first use.
    """

    def __init__(self) -> None:
        # Chunking
        chunker_map: dict[str, Any] = {
            "fixed": FixedSizeChunker(settings.chunk_size, settings.chunk_overlap),
            "recursive": RecursiveChunker(settings.chunk_size, settings.chunk_overlap),
            "semantic": SemanticChunker(max_chunk_tokens=settings.chunk_size),
        }
        self.chunker = chunker_map[settings.chunk_strategy]

        # Embeddings
        self.embedder = OpenAIEmbedder()

        # Vector store
        store_map: dict[str, Any] = {
            "pgvector": PgVectorStore,
            "pinecone": PineconeStore,
            "weaviate": WeaviateStore,
        }
        self.vector_store = store_map[settings.vector_store]()

        # Retrieval
        self.retriever = HybridRetriever(
            vector_store=self.vector_store,
            alpha=settings.hybrid_alpha,
            top_k=settings.top_k,
        )
        self.reranker: CrossEncoderReranker | None = (
            CrossEncoderReranker(top_n=settings.rerank_top_n) if settings.enable_reranking else None
        )
        self.context_builder = ContextBuilder()

        # Generation
        self.generator = OpenAIGenerator()

    async def ingest(
        self,
        text: str,
        source_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Chunk, embed, and store a plain-text document."""
        meta = metadata or {}
        chunks = self.chunker.chunk(text, meta, source_id)
        logger.info("pipeline_chunked", source_id=source_id, chunk_count=len(chunks))

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
        self.retriever.index_corpus(records)

        logger.info("pipeline_ingested", source_id=source_id, chunks_stored=len(records))
        return {"chunks_created": len(records)}

    async def query(
        self,
        query: str,
        top_k: int | None = None,
        enable_reranking: bool | None = None,
    ) -> dict[str, Any]:
        """Retrieve relevant chunks and generate a grounded answer."""
        if top_k is not None:
            self.retriever.top_k = top_k

        query_embedding = await asyncio.to_thread(self.embedder.embed, query)
        chunks = self.retriever.retrieve(query, query_embedding)
        logger.info("pipeline_retrieved", query=query[:80], chunk_count=len(chunks))

        use_reranking = enable_reranking if enable_reranking is not None else settings.enable_reranking
        if use_reranking and self.reranker:
            chunks = self.reranker.rerank(query, chunks)
            logger.info("pipeline_reranked", chunk_count=len(chunks))

        context = self.context_builder.build(query, chunks)
        answer = await asyncio.to_thread(
            self.generator.generate, query, context["context_text"]
        )

        return {
            "answer": answer,
            "sources": context["sources"],
            "retrieval_count": len(chunks),
        }
