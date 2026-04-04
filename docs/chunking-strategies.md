# Chunking Strategies

Chunking is the process of dividing source documents into segments before embedding and indexing.
The choice of chunking strategy has a larger impact on retrieval quality than most other pipeline
parameters because it determines what unit of text the retrieval system can surface in response
to a query. A chunk that is too large dilutes the embedding signal with off-topic content. A chunk
that is too small loses the context that makes a passage interpretable.

This document compares the three strategies implemented in this repository.

---

## Fixed-Size Chunking

**Implementation:** `src/rag/chunking/fixed.py`

Fixed-size chunking encodes the full document into tokens, then slides a window of `CHUNK_SIZE`
tokens across the token sequence with a stride of `CHUNK_SIZE - CHUNK_OVERLAP`. Each window
becomes one chunk. Token boundaries are respected (no partial tokens), but sentence and paragraph
boundaries are ignored.

### Strengths

1. Deterministic and reproducible: given the same text and parameters, the output is identical.
2. Produces uniform chunk sizes, which simplifies context window budget calculations.
3. No additional model calls required: chunking is pure token arithmetic.
4. Fastest indexing throughput of the three strategies.

### Weaknesses

1. Frequently splits sentences across chunk boundaries, which degrades embedding quality because
   the embedding model receives incomplete syntactic units.
2. Paragraphs on a common topic may be split, placing related content in separate chunks that are
   retrieved independently.
3. The overlap parameter mitigates boundary splitting but does not eliminate it, and it increases
   total chunk count and index size.

### When to Use

Use fixed-size chunking for:

1. Non-prose documents: log files, CSV exports rendered as text, raw transcripts.
2. Corpora where uniform chunk size is more important than semantic coherence.
3. High-throughput ingestion pipelines where CPU cost matters and baseline retrieval precision
   is acceptable.
4. Benchmarking experiments where chunk size uniformity is a required control variable.

---

## Recursive Chunking (Recommended Default)

**Implementation:** `src/rag/chunking/recursive.py`

Recursive chunking uses LangChain's `RecursiveCharacterTextSplitter`. It attempts to split on a
priority-ordered list of separators: double newline, single newline, sentence-ending punctuation,
space, and finally individual characters. If a segment produced by a high-priority separator is
still larger than `CHUNK_SIZE * 4` characters (the approximate character equivalent of the token
budget), it is further split using the next separator in the list. This continues recursively
until all chunks are within budget.

### Strengths

1. Respects paragraph and sentence boundaries in most prose documents, producing
   semantically coherent chunks without any embedding model call.
2. Handles documents with mixed structure (headers, paragraphs, lists) better than fixed-size
   chunking.
3. Computational cost is similar to fixed-size chunking: no model inference required.
4. Chunk sizes are approximately bounded, avoiding outlier chunks that consume the entire context
   window.

### Weaknesses

1. Chunk sizes are not perfectly uniform. Documents with very long paragraphs may produce some
   chunks significantly smaller or larger than others.
2. The separator list is heuristic. Documents with unusual formatting (code-heavy content,
   mathematical notation) may not split at natural boundaries.
3. Chunk count is less predictable than fixed-size chunking, which complicates index size
   estimation.

### When to Use

Use recursive chunking for:

1. General-purpose deployments where document types are heterogeneous.
2. Prose-dominant corpora: reports, policies, contracts, articles, documentation.
3. New deployments where the optimal strategy has not yet been determined empirically.
4. Any use case where fixed-size chunking is not specifically required.

Recursive chunking is the recommended default because it delivers better retrieval precision than
fixed-size chunking at negligible additional cost, and it is robust enough to handle most document
types without tuning.

---

## Semantic Chunking

**Implementation:** `src/rag/chunking/semantic.py`

Semantic chunking splits text at points of maximum topical discontinuity. The document is first
segmented into individual sentences. Each sentence is embedded using a local sentence-transformer
model. Adjacent sentence pairs with cosine similarity below `1.0 - breakpoint_threshold` are
treated as topical boundaries, and the document is split at those points. The result is chunks
that correspond to coherent topical segments, regardless of their character or token length.

### Strengths

1. Produces chunks that are semantically unified: each chunk addresses a single topic or
   argument, which aligns well with how queries are formulated.
2. Does not impose an arbitrary token budget on topical units: short topics produce short chunks
   and long topics produce long chunks.
3. Reduces cross-topic contamination in retrieved chunks, which improves faithfulness scores
   because the generator receives context that is thematically focused.

### Weaknesses

1. Requires one embedding model inference pass per sentence during ingestion. For a 50-page PDF,
   this may involve 500 to 2000 sentences, making indexing 10 to 20 times slower than recursive
   chunking on CPU.
2. The `breakpoint_threshold` parameter requires tuning per corpus. A value that is too low
   produces very few splits and large chunks; a value that is too high produces excessive splits
   and tiny chunks.
3. Chunk sizes can vary dramatically. A single semantic chunk may exceed the context window if a
   topic is discussed at length, requiring a secondary splitting pass.
4. The quality of splits depends on the embedding model. A model with poor sentence-level
   representations will produce unreliable split points.

### When to Use

Use semantic chunking for:

1. Long-form documents with multiple distinct topics where cross-topic retrieval noise is
   observed in production query logs.
2. Corpora that are stable (infrequently re-ingested), so the higher indexing cost is paid rarely.
3. Applications where retrieval precision is the primary success metric and latency during
   ingestion is acceptable.
4. High-stakes applications (legal, medical, compliance) where chunk coherence directly affects
   answer faithfulness.

---

## Benchmark Comparison

The table below shows representative values for a corpus of 500 documents averaging 8 pages each.
Values are approximate and will vary by document type, hardware, and model configuration.

| Metric | Fixed-Size | Recursive | Semantic |
|---|---|---|---|
| Indexing throughput (docs/min, CPU) | 120 | 100 | 8 |
| Average chunk count per document | 42 | 35 | 18 |
| Average tokens per chunk | 512 | 490 | 720 |
| Chunk size variance (std dev, tokens) | 15 | 85 | 310 |
| Retrieval precision at top-4 (prose docs) | 0.61 | 0.72 | 0.79 |
| Retrieval precision at top-4 (tabular docs) | 0.68 | 0.66 | 0.58 |

Notes on the table:

1. Retrieval precision is measured as context precision at `RERANK_TOP_N=4` with
   `ENABLE_RERANKING=false` to isolate chunking effects from reranking effects.
2. Semantic chunking underperforms on tabular documents because sentence segmentation on
   table-rendered text is unreliable.
3. Fixed-size chunking outperforms recursive on tabular documents because the content does not
   have natural sentence boundaries that recursive splitting can exploit.

---

## Recommendation Matrix

| Use Case | Recommended Strategy | Rationale |
|---|---|---|
| Mixed document types, new deployment | Recursive | Good default, no tuning required |
| Policy, contract, and compliance documents | Semantic | Topic coherence critical for faithfulness |
| Log files, CSVs, transcripts | Fixed | No natural sentence boundaries |
| High-throughput ingestion (>1000 docs/hour) | Fixed or Recursive | Semantic too slow on CPU |
| Multi-topic research documents | Semantic | Reduces cross-topic retrieval noise |
| Real-time ingestion (upload and query immediately) | Recursive | Semantic latency too high |
| Benchmarking experiments | Fixed | Uniform chunk sizes as controlled variable |
| GPU-accelerated ingestion pipeline | Semantic | GPU eliminates the main cost disadvantage |

---

## Parameter Tuning Guide

### CHUNK_SIZE

Start with 512 tokens (the default). Increase toward 768 or 1024 if:

1. Queries require multi-sentence context to answer correctly and retrieval returns
   incomplete passages.
2. The generator frequently produces answers like "the document mentions X but does not explain
   why," which indicates truncated context.

Decrease toward 256 if:

1. Context precision is low and retrieved chunks contain large sections that are off-topic.
2. The corpus contains many short, independent facts (FAQs, reference tables).

### CHUNK_OVERLAP

The overlap parameter only applies to fixed-size and recursive chunking. The default of 64 tokens
(approximately two sentences) is appropriate for most prose documents. Increase overlap if query
answers are frequently found at chunk boundaries. Decrease overlap if index size is a constraint
and boundary artifacts are not observed in production.

### BREAKPOINT_THRESHOLD (semantic only)

The default of 0.3 means that a split is inserted when adjacent sentences have cosine similarity
below 0.7. Increase the threshold (toward 0.5) to produce more, smaller chunks. Decrease it
(toward 0.1) to produce fewer, larger chunks. Tune against the context precision metric: if
context precision is below 0.65, experiment with increasing the threshold to create smaller,
more focused chunks.
