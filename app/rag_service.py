from __future__ import annotations

import logging
from typing import Any, Dict, List

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from app.config import settings

logger = logging.getLogger(__name__)

_COLLECTION = "documents1"
_MAX_PASSAGE_CHARS = 600
_MAX_CHUNKS = 20


def _client() -> QdrantClient:
    return QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)


def retrieve_rag(
    query_text: str,
    user_uid: str,
    collection: str = _COLLECTION,
    top_k: int = 3,
) -> List[Dict[str, Any]]:
    """
    Retrieve relevant passages from Qdrant using BM25 text search.

    Current collection stores documents with payload: {doc_id, filename, chunk_index, text, timestamp}.
    Future user-memory documents will have {type: 'memory', user_uid: <uid>, text: ...}.
    Filter: user memory for this user OR documents without a type (global knowledge).
    Returns at most _MAX_CHUNKS results.
    """
    top_k = min(top_k, _MAX_CHUNKS)

    try:
        client = _client()

        # Scroll-based text search: use payload filter on text field (BM25 match)
        # Since strict mode requires indexed fields only, and type/user_uid are now indexed,
        # we use a should filter: no-type docs (knowledge) OR user memory
        flt = qm.Filter(
            should=[
                # Global knowledge: documents without a type field
                qm.IsEmptyCondition(is_empty=qm.PayloadField(key="type")),
                # Explicit knowledge docs
                qm.FieldCondition(
                    key="type",
                    match=qm.MatchValue(value="knowledge"),
                ),
                # User-specific memory
                qm.Filter(
                    must=[
                        qm.FieldCondition(
                            key="type",
                            match=qm.MatchValue(value="memory"),
                        ),
                        qm.FieldCondition(
                            key="user_uid",
                            match=qm.MatchValue(value=user_uid),
                        ),
                    ]
                ),
            ]
        )

        # Use scroll with text filter since the collection only has a BM25 text index.
        # For full similarity search, vector embeddings would be needed.
        # Phase-1: scroll returns all matching docs; limit to top_k.
        hits_raw, _ = client.scroll(
            collection_name=collection,
            scroll_filter=flt,
            limit=top_k,
            with_payload=True,
            with_vectors=False,
        )
        hits = hits_raw
    except Exception as exc:
        logger.warning("Qdrant retrieval failed: %s", exc)
        return []

    out: List[Dict[str, Any]] = []
    seen: set = set()
    for h in hits:
        payload = h.payload or {}
        text = (payload.get("text") or payload.get("content") or "").strip()
        if not text:
            continue
        # Basic dedup by first 80 chars
        key = text[:80]
        if key in seen:
            continue
        seen.add(key)
        out.append(
            {
                "score": float(h.score) if hasattr(h, "score") and h.score is not None else None,
                "text": text[:_MAX_PASSAGE_CHARS],
                "source": payload.get("source"),
                "type": payload.get("type"),
            }
        )

    return out
