from __future__ import annotations

import json
from typing import Any

import psycopg2
import psycopg2.extras
from pgvector.psycopg2 import register_vector

from .base import BaseVectorStore
from ..config import settings


class PgVectorStore(BaseVectorStore):
    """
    PostgreSQL + pgvector vector store.

    Uses psycopg2 directly (no SQLAlchemy).
    Creates the pgvector extension and documents table on first use.
    Supports cosine, L2, and inner product distance metrics.
    Provides IVFFlat approximate nearest-neighbor index creation.
    """

    def __init__(self, dsn: str | None = None) -> None:
        self.dsn = dsn or settings.database_url
        self._conn: psycopg2.extensions.connection | None = None
        self._setup()

    def _connect(self) -> psycopg2.extensions.connection:
        if self._conn is None or self._conn.closed:
            self._conn = psycopg2.connect(self.dsn)
            register_vector(self._conn)
        return self._conn

    def _setup(self) -> None:
        conn = self._connect()
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    id          TEXT PRIMARY KEY,
                    content     TEXT NOT NULL,
                    embedding   vector(%s) NOT NULL,
                    metadata    JSONB DEFAULT '{}',
                    source_id   TEXT NOT NULL,
                    created_at  TIMESTAMPTZ DEFAULT now()
                );
                """,
                (settings.embedding_dimension,),
            )
        conn.commit()

    def create_ivfflat_index(self, lists: int = 100, metric: str = "cosine") -> None:
        """
        Create an IVFFlat approximate nearest-neighbor index.

        Args:
            lists: Number of IVF lists. Higher = better recall, slower build.
                   Rule of thumb: sqrt(num_rows) to 4 * sqrt(num_rows).
            metric: Distance metric -- 'cosine', 'l2', or 'inner_product'.
        """
        op_class = {
            "cosine": "vector_cosine_ops",
            "l2": "vector_l2_ops",
            "inner_product": "vector_ip_ops",
        }.get(metric, "vector_cosine_ops")
        conn = self._connect()
        with conn.cursor() as cur:
            cur.execute(
                f"""
                CREATE INDEX IF NOT EXISTS documents_embedding_ivfflat_idx
                ON documents USING ivfflat (embedding {op_class})
                WITH (lists = {lists});
                """
            )
        conn.commit()

    def upsert(self, records: list[dict[str, Any]]) -> None:
        conn = self._connect()
        with conn.cursor() as cur:
            psycopg2.extras.execute_values(
                cur,
                """
                INSERT INTO documents (id, content, embedding, metadata, source_id)
                VALUES %s
                ON CONFLICT (id) DO UPDATE
                    SET content    = EXCLUDED.content,
                        embedding  = EXCLUDED.embedding,
                        metadata   = EXCLUDED.metadata,
                        source_id  = EXCLUDED.source_id;
                """,
                [
                    (
                        r["id"],
                        r["content"],
                        r["embedding"],
                        json.dumps(r.get("metadata", {})),
                        r.get("source_id", ""),
                    )
                    for r in records
                ],
            )
        conn.commit()

    def similarity_search(
        self,
        embedding: list[float],
        top_k: int = 10,
        metric: str = "cosine",
    ) -> list[tuple[dict[str, Any], float]]:
        """
        Search for similar vectors.

        Args:
            embedding: Query vector.
            top_k: Number of results to return.
            metric: Distance metric -- 'cosine', 'l2', or 'inner_product'.
        """
        operator = {"cosine": "<=>", "l2": "<->", "inner_product": "<#>"}[metric]
        conn = self._connect()
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                f"""
                SELECT
                    id,
                    content,
                    metadata,
                    source_id,
                    1 - (embedding {operator} %s::vector) AS score
                FROM documents
                ORDER BY embedding {operator} %s::vector
                LIMIT %s;
                """,
                (embedding, embedding, top_k),
            )
            rows = cur.fetchall()
        return [
            (
                {
                    "id": row["id"],
                    "content": row["content"],
                    "metadata": row["metadata"] or {},
                    "source_id": row["source_id"],
                },
                float(row["score"]),
            )
            for row in rows
        ]

    def delete(self, ids: list[str]) -> None:
        conn = self._connect()
        with conn.cursor() as cur:
            cur.execute("DELETE FROM documents WHERE id = ANY(%s);", (ids,))
        conn.commit()
