# Architecture

This document describes the internal design of the enterprise-rag-production pipeline: how data
flows through the system, why each component was designed the way it was, and the trade-offs that
informed those decisions.

---

## Pipeline Stages and Data Flow

The pipeline has two distinct paths: an ingestion path that processes documents offline, and a
query path that executes on every user request.

### Ingestion Path

```
Raw Document
     |
     v
DocumentLoader          (src/rag/ingestion/loader.py)
  Supports: PDF, DOCX, HTML, TXT
  Output: raw text + source metadata
     |
     v
Preprocessor            (src/rag/ingestion/preprocessor.py)
  Normalizes whitespace, strips boilerplate
  Extracts structural metadata (title, page, section)
  Deduplicates by content hash before chunking
     |
     v
Chunker                 (src/rag/chunking/)
  Splits text into Chunk objects
  Each Chunk carries: content, metadata, chunk_index, source_id, token_count
     |
     v
Embedder                (src/rag/embeddings/)
  Converts chunk content to dense vectors
  Batched to respect API rate limits
  Caches by content hash to skip re-embedding unchanged text
     |
     v
VectorStore.upsert()    (src/rag/vectorstore/)
  Persists vector + metadata
  Also indexes raw text into in-memory BM25 corpus
```

The ingestion pipeline is orchestrated by `IngestionPipeline` in
`src/rag/ingestion/pipeline.py`. It runs synchronously to allow straightforward error handling
and retry logic. For large corpora, callers can batch documents and call the pipeline
concurrently at the document level.

### Query Path

```
POST /api/v1/query
  { query: str, top_k?: int, enable_reranking?: bool }
     |
     v
API Middleware           (src/rag/api/middleware.py)
  API key validation
  Request logging with correlation ID
  Prometheus request counter increment
     |
     v
Embedder.embed(query)
  Converts query string to dense vector
     |
     v
HybridRetriever.retrieve()    (src/rag/retrieval/retriever.py)
  Dense: VectorStore.similarity_search(query_vector, top_k * 2)
  Sparse: BM25Okapi.get_scores(tokenized_query)
  Fusion: normalized_score = alpha * dense + (1 - alpha) * bm25
  Returns top_k RetrievedChunk objects sorted by fused score
     |
     v
CrossEncoderReranker.rerank()  (src/rag/retrieval/reranker.py)
  Scores each (query, chunk) pair with a cross-encoder model
  Returns top_n RetrievedChunk objects sorted by cross-encoder score
  (skipped if ENABLE_RERANKING=false)
     |
     v
ContextBuilder.build()         (src/rag/retrieval/context_builder.py)
  Concatenates chunk content in ranked order
  Attaches source attribution metadata
  Enforces context window token budget
     |
     v
Generator.generate()           (src/rag/generation/openai_gen.py)
  Fills RAG prompt template with query and context
  Calls GPT-4o chat completions API
  Returns answer string
     |
     v
Response
  { answer, sources, query, retrieval_count }
```

---

## Chunking Strategy Selection

The chunking strategy controls how source documents are divided before embedding. The choice
affects retrieval precision, indexing throughput, and chunk count at a given document size.

### When to Use Fixed-Size Chunking

Fixed-size chunking splits text into non-overlapping windows of exactly `CHUNK_SIZE` tokens with
`CHUNK_OVERLAP` tokens of sliding overlap. It makes no attempt to respect sentence or paragraph
boundaries.

Use fixed-size chunking when:

1. Documents have no natural prose structure (tabular data, transcripts, log dumps).
2. Indexing throughput is the primary constraint and chunks do not need to be human-readable.
3. The corpus is homogeneous in structure and a uniform chunk size is known to work from prior
   experimentation.

Fixed-size chunking is the fastest strategy and produces the most predictable chunk counts, but
tends to fragment sentences, which degrades retrieval precision on prose documents.

### When to Use Recursive Chunking (Recommended Default)

Recursive chunking uses LangChain's `RecursiveCharacterTextSplitter`, which iteratively splits on
paragraph breaks, then sentence breaks, then word boundaries. It respects natural text structure
while still enforcing an approximate token budget per chunk.

Use recursive chunking for:

1. General-purpose deployments where document types are mixed or unknown.
2. Prose documents (reports, articles, policies, manuals) where sentence integrity matters.
3. New deployments where the optimal strategy has not yet been determined empirically.

Recursive chunking is the recommended default because it produces better retrieval precision than
fixed-size chunking with a negligible throughput penalty.

### When to Use Semantic Chunking

Semantic chunking embeds each sentence individually, then identifies split points where cosine
similarity between adjacent sentence embeddings drops below a threshold. The result is chunks that
correspond to coherent topical segments rather than arbitrary token windows.

Use semantic chunking when:

1. Documents contain multiple distinct topics and cross-topic retrieval noise is a known problem.
2. Chunk coherence is more important than indexing speed.
3. The document corpus is stable (semantic chunking is expensive to re-run at scale on changing
   corpora).

Semantic chunking requires one embedding model pass per sentence during ingestion. For a 50-page
document, this is roughly 10 to 20 times slower than recursive chunking. It is not appropriate
for high-throughput ingestion pipelines without dedicated GPU inference.

---

## Hybrid Retrieval Design Rationale

### The Failure Modes of Dense-Only Retrieval

Dense retrieval encodes both query and document chunks into a shared embedding space and retrieves
by nearest-neighbor search. This works well for semantic matching: a query about "revenue growth"
will correctly retrieve a chunk discussing "increasing sales figures." However, dense retrieval
performs poorly on:

1. Named entities, product codes, and identifiers that the embedding model has not seen in
   training.
2. Negation queries, where semantically similar text is returned regardless of the negative
   qualifier.
3. Rare technical terminology where the model's embedding is imprecise.

### The Failure Modes of Sparse-Only Retrieval

BM25 and other term-frequency methods return documents that share tokens with the query. This
handles exact-match cases that dense retrieval misses, but fails when:

1. The user query uses different vocabulary than the document (synonym gap).
2. The answer is conceptually related but does not share surface-level terms.

### Fusion Scoring

The hybrid retriever normalizes both score distributions to [0, 1] and combines them linearly:

```
fused_score = alpha * (dense_score / max_dense) + (1 - alpha) * (bm25_score / max_bm25)
```

The default `alpha = 0.7` reflects the general finding that dense retrieval is more useful than
BM25 on natural language queries. Teams working with corpora heavy in identifiers, codes, or
specialized terminology should lower alpha toward 0.5 or below. The optimal value should be
determined by running the evaluation harness across a range of alpha values.

---

## Reranking Trade-offs

### Bi-encoder vs Cross-encoder

The retrieval step uses bi-encoders: query and document are embedded independently, and similarity
is computed by vector arithmetic. This is efficient at scale because document embeddings are
precomputed and stored. However, bi-encoders cannot model fine-grained query-document interaction
because the two representations are computed in isolation.

Cross-encoders jointly encode the query and document in a single forward pass, attending to both
simultaneously. This produces substantially better relevance scores at the cost of requiring one
forward pass per candidate. Cross-encoders cannot be precomputed, so they are only practical as
a reranking step over a small candidate set, not over the full corpus.

### Latency Budget

The reranker runs a cross-encoder inference call for each of the `TOP_K` candidates. With the
default `TOP_K=10` and the `cross-encoder/ms-marco-MiniLM-L-6-v2` model, reranking adds
approximately 50 to 150 milliseconds on CPU. On GPU, this is typically under 20 milliseconds.

If end-to-end latency under 200 milliseconds is required, consider:

1. Setting `ENABLE_RERANKING=false` and increasing `RERANK_TOP_N` to equal `TOP_K`.
2. Reducing `TOP_K` to 5 before enabling reranking.
3. Deploying the cross-encoder on a GPU instance.

### Precision Gain

In internal benchmarks on prose document corpora, reranking with a cross-encoder improves context
precision by 10 to 25 percentage points at `RERANK_TOP_N=4` relative to using the top-4
bi-encoder results directly. The gain is larger when the initial retrieval pool contains
topically adjacent but off-topic chunks.

---

## Vector Store Selection Guide

### pgvector - Self-hosted, SQL-native

pgvector extends PostgreSQL with a vector data type and approximate nearest-neighbor index
operators. It is appropriate when:

1. The team already operates a PostgreSQL instance with available capacity.
2. The vector corpus is expected to remain under 10 million documents.
3. Metadata filtering with SQL predicates is required (pgvector supports `WHERE` clause
   filtering on any column in the same table).
4. Operational simplicity is valued over horizontal scalability.

The pgvector adapter in this repository creates the `vector` extension, the `chunks` table, and
an IVFFlat index automatically on first use. No manual schema setup is required.

### Pinecone - Fully managed, horizontally scalable

Pinecone is a purpose-built managed vector database. It is appropriate when:

1. The corpus exceeds what a single PostgreSQL instance can serve with sub-100-millisecond
   query latency.
2. The team does not want to own vector index maintenance, replication, or scaling.
3. Serverless billing is preferred over always-on infrastructure.

The Pinecone adapter requires a pre-created index configured with the correct dimension and
distance metric. All other lifecycle management is handled by the Pinecone service.

### Weaviate - Self-hosted, schema-rich

Weaviate provides a native object store with vector indexing, multi-tenancy, and a GraphQL query
interface. It is appropriate when:

1. The corpus requires structured metadata filtering that goes beyond simple SQL predicates.
2. Multi-tenant isolation is a hard requirement (Weaviate supports per-tenant vector spaces).
3. The team prefers a self-hosted, open-source stack over a managed service.

The Weaviate service is included in `docker-compose.yml` and can be used for local development
without any external accounts.

---

## Evaluation Framework Design

The evaluation framework is designed around three properties: it must be runnable without human
annotation, it must produce scores that correlate with real user satisfaction, and it must be
fast enough to include in a CI pipeline.

### Faithfulness

Faithfulness is the most important metric for production RAG systems because it directly measures
hallucination. The metric is computed by prompting `gpt-4o-mini` to act as a judge: given the
retrieved context and the generated answer, it assigns a score representing the fraction of
factual claims in the answer that are explicitly supported by the context. Using a small, fast
model (gpt-4o-mini rather than gpt-4o) keeps evaluation cost low.

### Context Precision

Context precision is computed without any LLM call, using only embedding similarity. It measures
whether the retrieval step is returning relevant chunks or padding the context with noise. A
precision below 0.6 typically indicates a misconfigured retrieval step (wrong chunk size, wrong
alpha value, or insufficient BM25 corpus coverage) rather than a generation problem.

### Answer Relevance

Answer relevance is the simplest metric: it is the cosine similarity between the query embedding
and the answer embedding. It catches cases where the generator produces a factually grounded but
off-topic response, which can happen when the retrieved context contains tangentially related
information and the prompt template does not enforce topical focus.

### Synthetic Dataset Generation

The evaluation dataset in `src/rag/evaluation/dataset.py` generates query-answer pairs by
prompting the LLM to produce questions from sample context passages. This produces a plausible
dataset without manual annotation, but it has known biases: the generated questions tend to be
answerable from the context by construction, which inflates faithfulness scores relative to real
user queries. Teams should supplement the synthetic dataset with queries from production query
logs as soon as those become available.
