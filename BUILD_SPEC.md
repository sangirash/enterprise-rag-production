# Build Specification: enterprise-rag-production

## Overview

Build a production-grade Retrieval-Augmented Generation (RAG) pipeline as a reference implementation.
This is NOT a tutorial project. It is an opinionated, production-ready reference that demonstrates
enterprise-level architectural decisions, evaluation frameworks, and operational concerns.

Author: Sudarshan Angirash (https://angirash.in)
Language: Python 3.11+
License: MIT

---

## Project Structure to Create

```
enterprise-rag-production/
├── README.md
├── LICENSE
├── pyproject.toml
├── .env.example
├── .gitignore
├── docker-compose.yml
├── Makefile
├── docs/
│   ├── architecture.md
│   ├── chunking-strategies.md
│   ├── vector-store-selection.md
│   └── evaluation-guide.md
├── src/
│   └── rag/
│       ├── __init__.py
│       ├── pipeline.py              # Main RAGPipeline orchestrator
│       ├── config.py                # Pydantic settings
│       ├── chunking/
│       │   ├── __init__.py
│       │   ├── base.py              # Abstract chunker
│       │   ├── fixed.py             # Fixed-size chunker
│       │   ├── semantic.py          # Semantic chunker (sentence-transformers)
│       │   └── recursive.py        # Recursive character text splitter
│       ├── ingestion/
│       │   ├── __init__.py
│       │   ├── loader.py            # Document loader (PDF, TXT, HTML, DOCX)
│       │   ├── preprocessor.py      # Cleaning, deduplication, metadata extraction
│       │   └── pipeline.py         # Ingestion orchestration
│       ├── embeddings/
│       │   ├── __init__.py
│       │   ├── base.py              # Abstract embedder
│       │   ├── openai_embedder.py   # OpenAI text-embedding-3-small/large
│       │   └── local_embedder.py    # sentence-transformers local model
│       ├── vectorstore/
│       │   ├── __init__.py
│       │   ├── base.py              # Abstract vector store
│       │   ├── pgvector.py          # PostgreSQL + pgvector
│       │   ├── pinecone.py          # Pinecone
│       │   └── weaviate.py         # Weaviate
│       ├── retrieval/
│       │   ├── __init__.py
│       │   ├── retriever.py         # Hybrid retriever (dense + sparse BM25)
│       │   ├── reranker.py          # Cross-encoder reranking
│       │   └── context_builder.py  # Context window assembly with metadata
│       ├── generation/
│       │   ├── __init__.py
│       │   ├── base.py              # Abstract generator
│       │   ├── openai_gen.py        # OpenAI GPT-4o generation
│       │   └── prompt_templates.py  # Versioned prompt templates
│       ├── evaluation/
│       │   ├── __init__.py
│       │   ├── metrics.py           # Faithfulness, relevance, context precision
│       │   ├── harness.py           # Evaluation test harness
│       │   └── dataset.py          # Synthetic eval dataset generation
│       ├── observability/
│       │   ├── __init__.py
│       │   ├── tracing.py           # OpenTelemetry tracing
│       │   └── metrics.py          # Prometheus metrics instrumentation
│       └── api/
│           ├── __init__.py
│           ├── main.py              # FastAPI application
│           ├── routes/
│           │   ├── query.py         # POST /query
│           │   └── ingest.py        # POST /ingest
│           └── middleware.py        # Auth, rate limiting, logging
├── tests/
│   ├── unit/
│   │   ├── test_chunking.py
│   │   ├── test_retrieval.py
│   │   └── test_pipeline.py
│   └── integration/
│       └── test_end_to_end.py
└── scripts/
    ├── ingest_sample.py
    └── run_eval.py
```

---

## File Contents to Implement

### pyproject.toml

```toml
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.backends.legacy:build"

[project]
name = "enterprise-rag-production"
version = "0.1.0"
description = "Production-grade RAG pipeline reference implementation"
readme = "README.md"
requires-python = ">=3.11"
license = {text = "MIT"}
authors = [
  {name = "Sudarshan Angirash", email = "sudaangi@techtonicis.com"}
]
dependencies = [
    "fastapi>=0.115.0",
    "uvicorn[standard]>=0.30.0",
    "pydantic>=2.7.0",
    "pydantic-settings>=2.3.0",
    "openai>=1.40.0",
    "langchain>=0.2.0",
    "langchain-openai>=0.1.0",
    "langchain-community>=0.2.0",
    "sentence-transformers>=3.0.0",
    "rank-bm25>=0.2.2",
    "pgvector>=0.3.0",
    "psycopg2-binary>=2.9.9",
    "pinecone-client>=3.2.0",
    "weaviate-client>=4.7.0",
    "pypdf>=4.2.0",
    "python-docx>=1.1.0",
    "beautifulsoup4>=4.12.3",
    "tiktoken>=0.7.0",
    "opentelemetry-sdk>=1.25.0",
    "opentelemetry-exporter-otlp>=1.25.0",
    "prometheus-client>=0.20.0",
    "python-dotenv>=1.0.1",
    "structlog>=24.2.0",
    "httpx>=0.27.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.2.0",
    "pytest-asyncio>=0.23.0",
    "pytest-cov>=5.0.0",
    "ruff>=0.5.0",
    "mypy>=1.10.0",
]

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.mypy]
python_version = "3.11"
strict = true
```

### src/rag/config.py

```python
from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Literal


class Settings(BaseSettings):
    # LLM
    openai_api_key: str = Field(..., env="OPENAI_API_KEY")
    openai_model: str = Field("gpt-4o", env="OPENAI_MODEL")
    embedding_model: str = Field("text-embedding-3-small", env="EMBEDDING_MODEL")
    embedding_dimension: int = Field(1536, env="EMBEDDING_DIMENSION")

    # Vector store selection
    vector_store: Literal["pgvector", "pinecone", "weaviate"] = Field("pgvector", env="VECTOR_STORE")

    # pgvector
    database_url: str = Field("postgresql://rag:rag@localhost:5432/ragdb", env="DATABASE_URL")

    # Pinecone
    pinecone_api_key: str = Field("", env="PINECONE_API_KEY")
    pinecone_index: str = Field("rag-production", env="PINECONE_INDEX")

    # Weaviate
    weaviate_url: str = Field("http://localhost:8080", env="WEAVIATE_URL")

    # Chunking
    chunk_strategy: Literal["fixed", "semantic", "recursive"] = Field("recursive", env="CHUNK_STRATEGY")
    chunk_size: int = Field(512, env="CHUNK_SIZE")
    chunk_overlap: int = Field(64, env="CHUNK_OVERLAP")

    # Retrieval
    top_k: int = Field(10, env="TOP_K")
    rerank_top_n: int = Field(4, env="RERANK_TOP_N")
    enable_reranking: bool = Field(True, env="ENABLE_RERANKING")
    enable_hybrid_search: bool = Field(True, env="ENABLE_HYBRID_SEARCH")
    hybrid_alpha: float = Field(0.7, env="HYBRID_ALPHA")  # 1.0 = dense only, 0.0 = BM25 only

    # Observability
    otlp_endpoint: str = Field("http://localhost:4317", env="OTLP_ENDPOINT")
    prometheus_port: int = Field(9090, env="PROMETHEUS_PORT")

    # API
    api_key: str = Field("changeme", env="RAG_API_KEY")
    log_level: str = Field("INFO", env="LOG_LEVEL")

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()
```

### src/rag/chunking/base.py

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class Chunk:
    content: str
    metadata: dict[str, Any]
    chunk_index: int
    source_id: str
    token_count: int


class BaseChunker(ABC):
    @abstractmethod
    def chunk(self, text: str, metadata: dict[str, Any], source_id: str) -> list[Chunk]:
        pass
```

### src/rag/chunking/fixed.py

```python
import tiktoken
from typing import Any
from .base import BaseChunker, Chunk


class FixedSizeChunker(BaseChunker):
    """Fixed token-count chunker with configurable overlap."""

    def __init__(self, chunk_size: int = 512, overlap: int = 64, model: str = "gpt-4o"):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.encoder = tiktoken.encoding_for_model(model)

    def chunk(self, text: str, metadata: dict[str, Any], source_id: str) -> list[Chunk]:
        tokens = self.encoder.encode(text)
        chunks = []
        start = 0
        index = 0
        while start < len(tokens):
            end = min(start + self.chunk_size, len(tokens))
            chunk_tokens = tokens[start:end]
            chunk_text = self.encoder.decode(chunk_tokens)
            chunks.append(Chunk(
                content=chunk_text,
                metadata={**metadata, "chunk_strategy": "fixed"},
                chunk_index=index,
                source_id=source_id,
                token_count=len(chunk_tokens),
            ))
            start += self.chunk_size - self.overlap
            index += 1
        return chunks
```

### src/rag/chunking/recursive.py

```python
from typing import Any
from langchain.text_splitter import RecursiveCharacterTextSplitter
import tiktoken
from .base import BaseChunker, Chunk


class RecursiveChunker(BaseChunker):
    """Recursive character text splitter — respects paragraph and sentence boundaries."""

    def __init__(self, chunk_size: int = 512, overlap: int = 64):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size * 4,  # approximate char count
            chunk_overlap=overlap * 4,
            separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
        )
        self.encoder = tiktoken.encoding_for_model("gpt-4o")

    def chunk(self, text: str, metadata: dict[str, Any], source_id: str) -> list[Chunk]:
        splits = self.splitter.split_text(text)
        return [
            Chunk(
                content=split,
                metadata={**metadata, "chunk_strategy": "recursive"},
                chunk_index=i,
                source_id=source_id,
                token_count=len(self.encoder.encode(split)),
            )
            for i, split in enumerate(splits)
        ]
```

### src/rag/chunking/semantic.py

```python
from typing import Any
from sentence_transformers import SentenceTransformer
import numpy as np
from .base import BaseChunker, Chunk
import tiktoken


class SemanticChunker(BaseChunker):
    """Semantic chunker: splits at points of maximum embedding cosine dissimilarity."""

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        breakpoint_threshold: float = 0.3,
        max_chunk_tokens: int = 512,
    ):
        self.model = SentenceTransformer(model_name)
        self.threshold = breakpoint_threshold
        self.max_tokens = max_chunk_tokens
        self.encoder = tiktoken.encoding_for_model("gpt-4o")

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

    def chunk(self, text: str, metadata: dict[str, Any], source_id: str) -> list[Chunk]:
        sentences = [s.strip() for s in text.replace("\n", " ").split(". ") if s.strip()]
        if not sentences:
            return []

        embeddings = self.model.encode(sentences, convert_to_numpy=True)
        breakpoints = []
        for i in range(1, len(sentences)):
            sim = self._cosine_similarity(embeddings[i - 1], embeddings[i])
            if sim < (1.0 - self.threshold):
                breakpoints.append(i)

        chunks = []
        prev = 0
        for bp in breakpoints + [len(sentences)]:
            segment = ". ".join(sentences[prev:bp]) + "."
            tokens = self.encoder.encode(segment)
            chunks.append(Chunk(
                content=segment,
                metadata={**metadata, "chunk_strategy": "semantic"},
                chunk_index=len(chunks),
                source_id=source_id,
                token_count=len(tokens),
            ))
            prev = bp
        return chunks
```

### src/rag/retrieval/retriever.py

```python
from dataclasses import dataclass
from typing import Any
from rank_bm25 import BM25Okapi
import numpy as np


@dataclass
class RetrievedChunk:
    content: str
    metadata: dict[str, Any]
    source_id: str
    score: float
    retrieval_method: str


class HybridRetriever:
    """
    Combines dense vector search with BM25 sparse retrieval.
    Final score = alpha * dense_score + (1 - alpha) * bm25_score
    """

    def __init__(self, vector_store, alpha: float = 0.7, top_k: int = 10):
        self.vector_store = vector_store
        self.alpha = alpha
        self.top_k = top_k
        self._bm25: BM25Okapi | None = None
        self._corpus: list[dict] = []

    def index_corpus(self, chunks: list[dict]) -> None:
        self._corpus = chunks
        tokenized = [c["content"].lower().split() for c in chunks]
        self._bm25 = BM25Okapi(tokenized)

    def retrieve(self, query: str, query_embedding: list[float]) -> list[RetrievedChunk]:
        dense_results = self.vector_store.similarity_search(query_embedding, top_k=self.top_k * 2)

        if self._bm25 and self._corpus:
            bm25_scores = self._bm25.get_scores(query.lower().split())
            bm25_top_idx = np.argsort(bm25_scores)[::-1][: self.top_k * 2]
            bm25_results = [
                (self._corpus[i], float(bm25_scores[i])) for i in bm25_top_idx
            ]
        else:
            bm25_results = []

        return self._fuse(dense_results, bm25_results)

    def _fuse(
        self,
        dense: list[tuple[dict, float]],
        sparse: list[tuple[dict, float]],
    ) -> list[RetrievedChunk]:
        scores: dict[str, float] = {}
        chunks: dict[str, dict] = {}

        max_dense = max((s for _, s in dense), default=1.0) or 1.0
        for chunk, score in dense:
            cid = chunk["id"]
            scores[cid] = scores.get(cid, 0.0) + self.alpha * (score / max_dense)
            chunks[cid] = chunk

        max_sparse = max((s for _, s in sparse), default=1.0) or 1.0
        for chunk, score in sparse:
            cid = chunk["id"]
            scores[cid] = scores.get(cid, 0.0) + (1 - self.alpha) * (score / max_sparse)
            chunks[cid] = chunk

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[: self.top_k]
        return [
            RetrievedChunk(
                content=chunks[cid]["content"],
                metadata=chunks[cid].get("metadata", {}),
                source_id=chunks[cid].get("source_id", ""),
                score=score,
                retrieval_method="hybrid",
            )
            for cid, score in ranked
        ]
```

### src/rag/retrieval/reranker.py

```python
from sentence_transformers import CrossEncoder
from .retriever import RetrievedChunk
from dataclasses import replace


class CrossEncoderReranker:
    """
    Cross-encoder reranker for precision-optimised top-N selection.
    Uses a full query-document interaction model — slower but more accurate than bi-encoders.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", top_n: int = 4):
        self.model = CrossEncoder(model_name)
        self.top_n = top_n

    def rerank(self, query: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        if not chunks:
            return []
        pairs = [(query, c.content) for c in chunks]
        scores = self.model.predict(pairs)
        ranked = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)
        return [replace(c, score=float(s)) for c, s in ranked[: self.top_n]]
```

### src/rag/evaluation/metrics.py

```python
"""
Evaluation metrics for RAG pipelines.

Faithfulness: fraction of answer claims that are grounded in retrieved context.
Context Precision: fraction of retrieved chunks that are relevant to the query.
Answer Relevance: semantic similarity between query and generated answer.
"""
from openai import OpenAI
from sentence_transformers import SentenceTransformer
import numpy as np
from dataclasses import dataclass

client = OpenAI()
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
    """Ask the LLM whether each sentence in the answer is supported by the context."""
    context = "\n\n".join(context_chunks)
    prompt = (
        f"Context:\n{context}\n\n"
        f"Answer:\n{answer}\n\n"
        "For each factual claim in the answer, determine if it is directly supported "
        "by the context above. Return a score between 0.0 and 1.0 where 1.0 means "
        "all claims are fully supported. Return only the numeric score."
    )
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    try:
        return float(response.choices[0].message.content.strip())
    except ValueError:
        return 0.0


def evaluate_context_precision(query: str, context_chunks: list[str]) -> float:
    """Fraction of retrieved chunks that contain information relevant to the query."""
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
    sim = np.dot(embs[0], embs[1]) / (
        np.linalg.norm(embs[0]) * np.linalg.norm(embs[1]) + 1e-8
    )
    return float(sim)
```

### src/rag/api/main.py

```python
from fastapi import FastAPI, Depends, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import make_asgi_app
from contextlib import asynccontextmanager
from .routes import query, ingest
from ..config import settings
import structlog

logger = structlog.get_logger()


def verify_api_key(x_api_key: str = Header(...)):
    if x_api_key != settings.api_key:
        raise HTTPException(status_code=401, detail="Invalid API key")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("RAG API starting", vector_store=settings.vector_store)
    yield
    logger.info("RAG API shutting down")


app = FastAPI(
    title="Enterprise RAG API",
    description="Production-grade Retrieval-Augmented Generation pipeline",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.include_router(query.router, prefix="/api/v1", dependencies=[Depends(verify_api_key)])
app.include_router(ingest.router, prefix="/api/v1", dependencies=[Depends(verify_api_key)])

metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)


@app.get("/health")
async def health():
    return {"status": "ok", "vector_store": settings.vector_store}
```

### src/rag/api/routes/query.py

```python
from fastapi import APIRouter
from pydantic import BaseModel
from ...pipeline import RAGPipeline

router = APIRouter()
pipeline = RAGPipeline()


class QueryRequest(BaseModel):
    query: str
    top_k: int | None = None
    enable_reranking: bool | None = None


class QueryResponse(BaseModel):
    answer: str
    sources: list[dict]
    query: str
    retrieval_count: int


@router.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    result = await pipeline.query(
        query=request.query,
        top_k=request.top_k,
        enable_reranking=request.enable_reranking,
    )
    return QueryResponse(
        answer=result["answer"],
        sources=result["sources"],
        query=request.query,
        retrieval_count=result["retrieval_count"],
    )
```

### src/rag/api/routes/ingest.py

```python
from fastapi import APIRouter, UploadFile, File, Form
from pydantic import BaseModel
from ...ingestion.pipeline import IngestionPipeline

router = APIRouter()
ingestion = IngestionPipeline()


class IngestResponse(BaseModel):
    source_id: str
    chunks_created: int
    status: str


@router.post("/ingest", response_model=IngestResponse)
async def ingest_file(
    file: UploadFile = File(...),
    source_id: str = Form(...),
):
    content = await file.read()
    result = await ingestion.ingest(
        content=content,
        filename=file.filename or "unknown",
        source_id=source_id,
    )
    return IngestResponse(
        source_id=source_id,
        chunks_created=result["chunks_created"],
        status="success",
    )
```

### docker-compose.yml

```yaml
version: "3.9"

services:
  postgres:
    image: pgvector/pgvector:pg16
    environment:
      POSTGRES_USER: rag
      POSTGRES_PASSWORD: rag
      POSTGRES_DB: ragdb
    ports:
      - "5432:5432"
    volumes:
      - pgdata:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U rag -d ragdb"]
      interval: 10s
      timeout: 5s
      retries: 5

  weaviate:
    image: semitechnologies/weaviate:1.25.4
    ports:
      - "8080:8080"
    environment:
      QUERY_DEFAULTS_LIMIT: 25
      AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED: "true"
      PERSISTENCE_DATA_PATH: /var/lib/weaviate
      DEFAULT_VECTORIZER_MODULE: none
      ENABLE_MODULES: ""
      CLUSTER_HOSTNAME: node1

  rag-api:
    build: .
    ports:
      - "8000:8000"
      - "9090:9090"
    env_file: .env
    depends_on:
      postgres:
        condition: service_healthy
    volumes:
      - ./data:/app/data

  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9091:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      GF_SECURITY_ADMIN_PASSWORD: admin
    volumes:
      - grafana_data:/var/lib/grafana

volumes:
  pgdata:
  grafana_data:
```

### Makefile

```makefile
.PHONY: install dev test lint format build up down clean eval

install:
	pip install -e ".[dev]"

dev:
	uvicorn src.rag.api.main:app --reload --host 0.0.0.0 --port 8000

test:
	pytest tests/ -v --cov=src --cov-report=term-missing

lint:
	ruff check src/ tests/
	mypy src/

format:
	ruff format src/ tests/

build:
	docker build -t enterprise-rag:latest .

up:
	docker compose up -d

down:
	docker compose down

clean:
	docker compose down -v
	find . -type d -name __pycache__ | xargs rm -rf
	rm -rf .coverage htmlcov/ dist/ *.egg-info

eval:
	python scripts/run_eval.py
```

### .env.example

```
# LLM Provider
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_DIMENSION=1536

# Vector Store: pgvector | pinecone | weaviate
VECTOR_STORE=pgvector

# pgvector (default)
DATABASE_URL=postgresql://rag:rag@localhost:5432/ragdb

# Pinecone (if using Pinecone)
PINECONE_API_KEY=
PINECONE_INDEX=rag-production

# Weaviate (if using Weaviate)
WEAVIATE_URL=http://localhost:8080

# Chunking: fixed | semantic | recursive
CHUNK_STRATEGY=recursive
CHUNK_SIZE=512
CHUNK_OVERLAP=64

# Retrieval
TOP_K=10
RERANK_TOP_N=4
ENABLE_RERANKING=true
ENABLE_HYBRID_SEARCH=true
HYBRID_ALPHA=0.7

# Observability
OTLP_ENDPOINT=http://localhost:4317
PROMETHEUS_PORT=9090

# API
RAG_API_KEY=changeme
LOG_LEVEL=INFO
```

### .gitignore

```
__pycache__/
*.py[cod]
*.so
.env
.venv/
venv/
dist/
build/
*.egg-info/
.coverage
htmlcov/
.mypy_cache/
.ruff_cache/
.pytest_cache/
data/
*.db
*.sqlite
```

---

## README.md Requirements

Write a professional README with NO emoji and NO em dash characters (use regular hyphen - instead of -).
Use only standard ASCII. Professional tone, as if written for a senior engineering audience.

Structure:
1. Title: enterprise-rag-production
2. One-line description
3. Overview section (3-4 paragraphs explaining the problem and what this solves)
4. Architecture section with an ASCII diagram showing the pipeline flow
5. Key Design Decisions section (chunking strategies, hybrid retrieval, reranking, vector store abstraction)
6. Quickstart section (prerequisites, installation, docker compose up, first query)
7. Configuration Reference table (all env vars with types and defaults)
8. Vector Store Adapters section (pgvector, Pinecone, Weaviate - when to use each)
9. Evaluation Framework section (the three metrics, how to run eval)
10. Project Structure tree
11. Contributing guidelines
12. License

Rules for README:
- No emoji anywhere
- No em dash (--) - use a regular hyphen or restructure the sentence
- No bullet points with dashes - use numbered lists or restructure
- Use markdown headers, code blocks, and tables
- Professional engineering tone
- Author attribution at bottom: "Designed and implemented by Sudarshan Angirash (https://angirash.in)"

---

## docs/architecture.md

Write a detailed architecture document covering:
- Pipeline stages and data flow
- Chunking strategy selection criteria (when to use fixed vs semantic vs recursive)
- Hybrid retrieval design rationale (why dense + sparse, alpha tuning)
- Reranking trade-offs (latency vs precision)
- Vector store selection guide (pgvector for <10M docs, Pinecone for managed scale, Weaviate for schema-rich)
- Evaluation framework design

No emoji, no em dashes.

---

## docs/chunking-strategies.md

Write a technical comparison document:
- Fixed chunking: pros, cons, when to use
- Recursive chunking: pros, cons, when to use (recommended default)
- Semantic chunking: pros, cons, when to use
- Benchmark table (hypothetical but realistic): retrieval precision vs indexing speed vs chunk count
- Recommendation matrix by use case

---

## Additional Implementation Notes

1. src/rag/pipeline.py should be the main orchestrator that wires together: ingestion -> chunking -> embedding -> vector store -> retrieval -> reranking -> generation

2. src/rag/vectorstore/pgvector.py should:
   - Use psycopg2 directly (not SQLAlchemy)
   - Create the extension and table on first use
   - Support cosine, L2, and inner product distance metrics
   - Include an IVFFlat index creation method

3. src/rag/embeddings/openai_embedder.py should:
   - Batch embeddings (max 100 per request)
   - Handle rate limiting with exponential backoff
   - Cache embeddings by content hash to avoid redundant API calls

4. src/rag/generation/prompt_templates.py should define:
   - A RAG_SYSTEM_PROMPT constant
   - A build_rag_prompt(query, context_chunks, metadata) function
   - A FAITHFULNESS_CHECK_PROMPT constant

5. tests/unit/test_chunking.py should test all three chunkers with a sample 500-word document

6. scripts/ingest_sample.py should ingest a sample text file from a --path argument

7. scripts/run_eval.py should run the evaluation harness against a small synthetic dataset and print a results table

---

## NOTIFICATION

When completely done, run:
openclaw system event --text "enterprise-rag-production repo build complete: full RAG pipeline with eval harness ready for review" --mode now
