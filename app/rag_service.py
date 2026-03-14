from __future__ import annotations

import logging
from typing import Any, Dict, List

from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from app.config import settings

logger = logging.getLogger(__name__)

_COLLECTION = "documents1"
_MAX_PASSAGE_CHARS = 600
_EMBEDDING_MODEL = "text-embedding-3-small"

_qdrant: QdrantClient | None = None
_openai: OpenAI | None = None


def _get_qdrant() -> QdrantClient:
    global _qdrant
    if _qdrant is None:
        _qdrant = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)
    return _qdrant


def _get_openai() -> OpenAI:
    global _openai
    if _openai is None:
        _openai = OpenAI(api_key=settings.openai_api_key)
    return _openai


def _embed_query(text: str) -> List[float]:
    """Generate embedding vector for a query using OpenAI."""
    resp = _get_openai().embeddings.create(model=_EMBEDDING_MODEL, input=text)
    return resp.data[0].embedding


def retrieve_rag(
    query_text: str,
    user_uid: str,
    collection: str = _COLLECTION,
    top_k: int | None = None,
) -> List[Dict[str, Any]]:
    """
    Retrieve relevant passages from Qdrant using vector similarity search.

    Generates a query embedding via OpenAI text-embedding-3-small, then
    searches the collection with cosine similarity.
    top_k defaults to settings.qdrant_top_k (QDRANT_TOP_K env var).
    """
    if top_k is None:
        top_k = settings.qdrant_top_k

    try:
        query_vec = _embed_query(query_text)
    except Exception as exc:
        logger.warning("Embedding generation failed: %s", exc)
        return []

    try:
        client = _get_qdrant()

        # Filter: global knowledge (no type) OR explicit knowledge OR user memory
        flt = qm.Filter(
            should=[
                qm.IsEmptyCondition(is_empty=qm.PayloadField(key="type")),
                qm.FieldCondition(
                    key="type",
                    match=qm.MatchValue(value="knowledge"),
                ),
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

        response = client.query_points(
            collection_name=collection,
            query=query_vec,
            query_filter=flt,
            limit=top_k,
            with_payload=True,
        )
        hits = response.points if hasattr(response, "points") else response
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
        # Dedup by first 80 chars
        key = text[:80]
        if key in seen:
            continue
        seen.add(key)
        out.append(
            {
                "score": float(h.score) if hasattr(h, "score") and h.score is not None else None,
                "text": text[:_MAX_PASSAGE_CHARS],
                "source": payload.get("filename"),
                "type": payload.get("type"),
            }
        )

    return out
