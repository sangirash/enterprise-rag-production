# enterprise-rag-production

A production-grade Retrieval-Augmented Generation pipeline reference implementation in Python 3.11+.

---

## Overview

Retrieval-Augmented Generation has become the dominant pattern for grounding large language model
outputs in proprietary or frequently updated knowledge. Most open-source RAG examples are
tutorial-quality: they demonstrate the concept but omit the operational concerns that determine
whether a system survives production traffic. This repository exists to fill that gap.

This implementation makes explicit architectural choices at every layer: pluggable chunking
strategies with measurable trade-offs, hybrid retrieval that combines dense vector search with BM25
sparse retrieval, a cross-encoder reranking step that trades latency for precision, and a vector
store abstraction that supports three distinct backends without changing application logic. Each
choice is documented and reversible.

The evaluation framework runs three metrics (faithfulness, context precision, and answer relevance)
against a synthetic dataset, giving teams a regression harness before deploying prompt or retrieval
changes. Observability is wired in from the start: OpenTelemetry traces and Prometheus metrics are
emitted on every request, not added later as an afterthought.

This is not a starter template. It is an opinionated reference that reflects how a senior
engineering team would build a RAG system intended to handle real query volumes, evolving document
corpora, and the operational requirements of enterprise deployment.

---

## Architecture

```
Documents (PDF, DOCX, HTML, TXT)
         |
         v
  [ Ingestion Pipeline ]
    Loader -> Preprocessor -> Deduplication
         |
         v
  [ Chunking ]
    FixedSizeChunker | RecursiveChunker | SemanticChunker
         |
         v
  [ Embedding ]
    OpenAIEmbedder | LocalEmbedder (sentence-transformers)
         |
         v
  [ Vector Store ]
    pgvector | Pinecone | Weaviate
         |
    (indexed corpus also held in memory for BM25)
         |
         v
  [ Query Path ]
    POST /api/v1/query
         |
         v
  [ HybridRetriever ]
    Dense vector search + BM25 sparse search
    Reciprocal rank fusion: score = alpha * dense + (1 - alpha) * sparse
         |
         v
  [ CrossEncoderReranker ]  (optional, enabled by default)
    Full query-document interaction model
    Selects top_n from top_k candidates
         |
         v
  [ Context Builder ]
    Assembles retrieved chunks with source metadata
         |
         v
  [ Generation ]
    OpenAI GPT-4o with versioned prompt templates
         |
         v
  [ Response ]
    { answer, sources, retrieval_count }
```

---

## Key Design Decisions

### Chunking Strategies

Three strategies are provided because no single strategy is optimal across document types.
Fixed-size chunking is fast and predictable but breaks semantic units at arbitrary token
boundaries. Recursive chunking respects paragraph and sentence structure while remaining
computationally inexpensive, making it the recommended default. Semantic chunking uses embedding
cosine dissimilarity to detect topical boundaries, producing semantically coherent chunks at the
cost of requiring a model inference pass during ingestion. The strategy is selected at runtime via
the `CHUNK_STRATEGY` environment variable, so the same deployment can be reconfigured without code
changes.

### Hybrid Retrieval

Dense vector retrieval excels at semantic matching but misses exact-match queries involving
product names, identifiers, or rare terminology. BM25 sparse retrieval captures exact term
frequency but fails on paraphrase and synonym queries. Combining both with a tunable alpha weight
recovers the strengths of each method. The default `HYBRID_ALPHA=0.7` weights dense search more
heavily; lowering it toward 0.0 shifts influence toward BM25. Teams should tune alpha against their
evaluation dataset rather than relying on the default.

### Reranking

The initial retrieval step returns `TOP_K` candidates, optimized for recall. The cross-encoder
reranker then selects `RERANK_TOP_N` from those candidates using a full query-document interaction
model, optimized for precision. This two-stage approach avoids running the expensive cross-encoder
over the entire corpus while still achieving better ranking quality than a bi-encoder alone. The
reranker can be disabled via `ENABLE_RERANKING=false` when latency is the primary constraint.

### Vector Store Abstraction

All three vector store backends implement the same abstract interface. The application selects a
backend at startup based on the `VECTOR_STORE` environment variable. This means switching from a
local pgvector instance during development to a managed Pinecone index in production requires only
a configuration change. The abstraction exposes the operations the RAG pipeline actually needs:
upsert, similarity search, and deletion by source ID.

---

## Quickstart

**Prerequisites:**

1. Python 3.11 or higher
2. Docker and Docker Compose
3. An OpenAI API key

**Installation:**

```bash
git clone https://github.com/sudaangi/enterprise-rag-production
cd enterprise-rag-production
cp .env.example .env
# Edit .env and set OPENAI_API_KEY
make install
```

**Start infrastructure:**

```bash
make up
# Starts PostgreSQL with pgvector, Weaviate, Prometheus, and Grafana
```

**Start the API:**

```bash
make dev
# API available at http://localhost:8000
# Metrics at http://localhost:8000/metrics
# Interactive docs at http://localhost:8000/docs
```

**Ingest a document:**

```bash
curl -X POST http://localhost:8000/api/v1/ingest \
  -H "X-API-Key: changeme" \
  -F "file=@/path/to/document.pdf" \
  -F "source_id=doc-001"
```

**Run a query:**

```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H "X-API-Key: changeme" \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the key findings in the document?"}'
```

**Run the evaluation harness:**

```bash
make eval
```

---

## Configuration Reference

| Variable | Type | Default | Description |
|---|---|---|---|
| `OPENAI_API_KEY` | string | required | OpenAI API key |
| `OPENAI_MODEL` | string | `gpt-4o` | Chat completion model |
| `EMBEDDING_MODEL` | string | `text-embedding-3-small` | Embedding model name |
| `EMBEDDING_DIMENSION` | integer | `1536` | Embedding vector dimension |
| `VECTOR_STORE` | enum | `pgvector` | Backend: `pgvector`, `pinecone`, or `weaviate` |
| `DATABASE_URL` | string | `postgresql://rag:rag@localhost:5432/ragdb` | PostgreSQL connection string |
| `PINECONE_API_KEY` | string | `""` | Pinecone API key (required if using Pinecone) |
| `PINECONE_INDEX` | string | `rag-production` | Pinecone index name |
| `WEAVIATE_URL` | string | `http://localhost:8080` | Weaviate HTTP endpoint |
| `CHUNK_STRATEGY` | enum | `recursive` | Strategy: `fixed`, `recursive`, or `semantic` |
| `CHUNK_SIZE` | integer | `512` | Target chunk size in tokens |
| `CHUNK_OVERLAP` | integer | `64` | Token overlap between adjacent chunks |
| `TOP_K` | integer | `10` | Candidate count retrieved before reranking |
| `RERANK_TOP_N` | integer | `4` | Final chunk count passed to the generator |
| `ENABLE_RERANKING` | boolean | `true` | Whether to apply cross-encoder reranking |
| `ENABLE_HYBRID_SEARCH` | boolean | `true` | Whether to combine dense and BM25 retrieval |
| `HYBRID_ALPHA` | float | `0.7` | Blend weight: 1.0 = dense only, 0.0 = BM25 only |
| `OTLP_ENDPOINT` | string | `http://localhost:4317` | OpenTelemetry collector endpoint |
| `PROMETHEUS_PORT` | integer | `9090` | Port for Prometheus metrics server |
| `RAG_API_KEY` | string | `changeme` | API key required in `X-API-Key` request header |
| `LOG_LEVEL` | string | `INFO` | Log level: `DEBUG`, `INFO`, `WARNING`, or `ERROR` |

---

## Vector Store Adapters

### pgvector

PostgreSQL with the pgvector extension. Appropriate for corpora up to approximately 10 million
vectors on a single node. Requires no external managed service and co-locates relational metadata
with vector data. The adapter creates the extension, table, and IVFFlat index automatically on
first use. Choose pgvector when the team already operates PostgreSQL and the document corpus fits
within a single database instance.

### Pinecone

Fully managed vector database with horizontal scaling handled by the provider. Appropriate when
the corpus exceeds what a single PostgreSQL instance can serve with acceptable query latency, or
when the team does not want to operate vector infrastructure. The Pinecone adapter requires an API
key and a pre-created index. Choose Pinecone for production deployments where operational
simplicity is valued over infrastructure control.

### Weaviate

Open-source vector database with a schema-rich data model, built-in object storage, and
multi-tenancy support. Appropriate when the corpus requires structured metadata filtering alongside
vector search, or when a self-hosted alternative to Pinecone is preferred. The Weaviate service
is included in `docker-compose.yml` for local development. Choose Weaviate when metadata filtering
and multi-tenant isolation are first-class requirements.

---

## Evaluation Framework

The evaluation framework measures three independent metrics on each query-answer pair.

**Faithfulness** measures what fraction of factual claims in the generated answer are grounded in
the retrieved context. It is computed by prompting `gpt-4o-mini` to score the answer against the
context. A score of 1.0 means every claim is supported; a score approaching 0.0 indicates
hallucination.

**Context Precision** measures what fraction of the retrieved chunks contain information relevant
to the query. It is computed using embedding cosine similarity between the query and each chunk,
with a relevance threshold of 0.4. Low context precision indicates that the retrieval step is
returning noise that the generator must ignore.

**Answer Relevance** measures semantic similarity between the query and the generated answer,
computed as the cosine similarity of their embeddings. Low answer relevance indicates that the
generator produced a response that does not address the question asked.

The overall score is the arithmetic mean of all three metrics.

```bash
make eval
# Prints a table with per-query scores and aggregate statistics
```

The synthetic evaluation dataset is generated by `src/rag/evaluation/dataset.py`. Teams are
encouraged to replace or supplement it with queries derived from real user traffic.

---

## Project Structure

```
enterprise-rag-production/
├── src/rag/
│   ├── pipeline.py              # Main RAGPipeline orchestrator
│   ├── config.py                # Pydantic settings loaded from environment
│   ├── api/
│   │   ├── main.py              # FastAPI application with lifespan and middleware
│   │   ├── middleware.py        # Authentication, rate limiting, request logging
│   │   └── routes/
│   │       ├── query.py         # POST /api/v1/query
│   │       └── ingest.py        # POST /api/v1/ingest
│   ├── chunking/
│   │   ├── base.py              # Chunk dataclass and BaseChunker abstract class
│   │   ├── fixed.py             # Token-counted fixed-size chunker
│   │   ├── recursive.py         # Recursive character text splitter
│   │   └── semantic.py          # Embedding-based semantic boundary detection
│   ├── embeddings/
│   │   ├── base.py              # Abstract embedder interface
│   │   ├── openai_embedder.py   # OpenAI embeddings with batching and caching
│   │   └── local_embedder.py    # sentence-transformers local inference
│   ├── vectorstore/
│   │   ├── base.py              # Abstract vector store interface
│   │   ├── pgvector.py          # PostgreSQL + pgvector adapter
│   │   ├── pinecone.py          # Pinecone adapter
│   │   └── weaviate.py          # Weaviate adapter
│   ├── retrieval/
│   │   ├── retriever.py         # HybridRetriever combining dense and BM25
│   │   ├── reranker.py          # CrossEncoderReranker
│   │   └── context_builder.py   # Context window assembly with source attribution
│   ├── generation/
│   │   ├── base.py              # Abstract generator interface
│   │   ├── openai_gen.py        # GPT-4o generation with structured output
│   │   └── prompt_templates.py  # Versioned system and user prompt templates
│   ├── evaluation/
│   │   ├── metrics.py           # Faithfulness, context precision, answer relevance
│   │   ├── harness.py           # Evaluation test harness
│   │   └── dataset.py           # Synthetic evaluation dataset generation
│   └── observability/
│       ├── tracing.py           # OpenTelemetry span instrumentation
│       └── metrics.py           # Prometheus counters, histograms, and gauges
├── tests/
│   ├── unit/
│   │   ├── test_chunking.py
│   │   ├── test_retrieval.py
│   │   └── test_pipeline.py
│   └── integration/
│       └── test_end_to_end.py
├── scripts/
│   ├── ingest_sample.py
│   └── run_eval.py
├── docs/
│   ├── architecture.md
│   └── chunking-strategies.md
├── docker-compose.yml
├── Makefile
├── pyproject.toml
└── .env.example
```

---

## Contributing

1. Fork the repository and create a feature branch from `main`.
2. Install development dependencies with `make install`.
3. Run `make lint` and `make test` before submitting a pull request. Both must pass with no
   errors or warnings.
4. Keep changes focused. Each pull request should address a single concern: a bug fix, a new
   adapter, a metric, or a documentation correction.
5. If adding a new vector store adapter, implement the full `BaseVectorStore` interface and add
   it to the adapter selection logic in `config.py`.
6. If changing retrieval or generation behavior, run `make eval` and include the before/after
   metric table in the pull request description.

---

## License

MIT License. See [LICENSE](LICENSE) for full terms.

---

Designed and implemented by Sudarshan Angirash (https://angirash.in)
