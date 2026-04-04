"""
Prometheus metrics instrumentation for the RAG pipeline.

Import this module once at startup to register all metrics.
Metrics are exposed via the /metrics endpoint mounted in the FastAPI app.
"""
from prometheus_client import Counter, Gauge, Histogram

query_total = Counter(
    "rag_query_total",
    "Total number of queries processed",
    ["status"],  # labels: success | error
)

query_latency_seconds = Histogram(
    "rag_query_latency_seconds",
    "End-to-end query latency in seconds",
    buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
)

retrieval_latency_seconds = Histogram(
    "rag_retrieval_latency_seconds",
    "Vector store + BM25 retrieval latency in seconds",
    buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.0],
)

reranking_latency_seconds = Histogram(
    "rag_reranking_latency_seconds",
    "Cross-encoder reranking latency in seconds",
    buckets=[0.05, 0.1, 0.25, 0.5, 1.0],
)

generation_latency_seconds = Histogram(
    "rag_generation_latency_seconds",
    "LLM generation latency in seconds",
    buckets=[0.5, 1.0, 2.5, 5.0, 10.0, 30.0],
)

chunks_retrieved = Histogram(
    "rag_chunks_retrieved",
    "Number of chunks retrieved per query",
    buckets=[1, 2, 4, 8, 16, 32],
)

ingestion_total = Counter(
    "rag_ingestion_total",
    "Total documents ingested",
    ["status"],  # labels: success | skipped | error
)

documents_indexed = Gauge(
    "rag_documents_indexed_total",
    "Cumulative number of document chunks in the vector store",
)

embedding_cache_hits = Counter(
    "rag_embedding_cache_hits_total",
    "Number of embedding requests served from the in-process cache",
)
