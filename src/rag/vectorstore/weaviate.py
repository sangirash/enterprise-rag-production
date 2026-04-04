from __future__ import annotations

import json
import uuid
from typing import Any

from .base import BaseVectorStore
from ..config import settings

_CLASS_NAME = "Document"


class WeaviateStore(BaseVectorStore):
    """
    Weaviate vector store adapter (weaviate-client v4).

    Uses a single 'Document' collection with no built-in vectorizer;
    vectors are supplied externally from our embedding module.
    """

    def __init__(self) -> None:
        import weaviate  # type: ignore[import]

        host_port = settings.weaviate_url.replace("http://", "").replace("https://", "")
        parts = host_port.split(":")
        host = parts[0]
        port = int(parts[1]) if len(parts) > 1 else 8080
        self.client = weaviate.connect_to_local(host=host, port=port)
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        from weaviate.classes.config import Configure, Property, DataType  # type: ignore[import]

        if not self.client.collections.exists(_CLASS_NAME):
            self.client.collections.create(
                _CLASS_NAME,
                vectorizer_config=Configure.Vectorizer.none(),
                properties=[
                    Property(name="content", data_type=DataType.TEXT),
                    Property(name="source_id", data_type=DataType.TEXT),
                    Property(name="doc_metadata", data_type=DataType.TEXT),
                ],
            )

    def _stable_uuid(self, doc_id: str) -> str:
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, doc_id))

    def upsert(self, records: list[dict[str, Any]]) -> None:
        collection = self.client.collections.get(_CLASS_NAME)
        with collection.batch.dynamic() as batch:
            for r in records:
                batch.add_object(
                    properties={
                        "content": r["content"],
                        "source_id": r.get("source_id", ""),
                        "doc_metadata": json.dumps(r.get("metadata", {})),
                    },
                    vector=r["embedding"],
                    uuid=self._stable_uuid(r["id"]),
                )

    def similarity_search(
        self,
        embedding: list[float],
        top_k: int = 10,
    ) -> list[tuple[dict[str, Any], float]]:
        collection = self.client.collections.get(_CLASS_NAME)
        result = collection.query.near_vector(
            near_vector=embedding,
            limit=top_k,
            return_metadata=["certainty"],
        )
        return [
            (
                {
                    "id": str(obj.uuid),
                    "content": obj.properties.get("content", ""),
                    "metadata": json.loads(obj.properties.get("doc_metadata", "{}")),
                    "source_id": obj.properties.get("source_id", ""),
                },
                float(obj.metadata.certainty or 0.0),
            )
            for obj in result.objects
        ]

    def delete(self, ids: list[str]) -> None:
        collection = self.client.collections.get(_CLASS_NAME)
        for doc_id in ids:
            collection.data.delete_by_id(self._stable_uuid(doc_id))
