"""
Document preprocessing: cleaning, deduplication, and metadata enrichment.

Deduplication uses a content hash so that re-ingesting the same document
does not create duplicate entries in the vector store.
"""
from __future__ import annotations

import hashlib
import re

from .loader import Document

# Process-lifetime set of content hashes for in-memory deduplication.
_seen_hashes: set[str] = set()


def preprocess(doc: Document, dedup: bool = True) -> Document | None:
    """
    Clean and deduplicate a document.

    Returns None if the document is empty or a duplicate (when dedup=True).
    """
    text = _clean(doc.content)
    if not text:
        return None

    content_hash = hashlib.sha256(text.encode()).hexdigest()
    if dedup and content_hash in _seen_hashes:
        return None
    _seen_hashes.add(content_hash)

    enriched_metadata = {
        **doc.metadata,
        "content_hash": content_hash,
        "word_count": len(text.split()),
        "char_count": len(text),
    }
    return Document(content=text, metadata=enriched_metadata, source_id=doc.source_id)


def reset_dedup_cache() -> None:
    """Clear the in-memory deduplication cache. Useful in tests."""
    _seen_hashes.clear()


def _clean(text: str) -> str:
    text = re.sub(r"\r\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\x00", "", text)
    return text.strip()
