"""
Test Qdrant vector similarity search on documents1 collection.
Generates query embedding via OpenAI text-embedding-3-small, then searches.
"""
from __future__ import annotations

import os
import sys

from dotenv import load_dotenv

load_dotenv()

from openai import OpenAI
from qdrant_client import QdrantClient

QDRANT_URL = os.getenv("QDRANT_URL", "")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
COLLECTION = "documents1"
TOP_K = int(os.getenv("QDRANT_TOP_K", "10"))


def get_embedding(text: str) -> list[float]:
    """Get OpenAI embedding for a query."""
    client = OpenAI(api_key=OPENAI_API_KEY)
    resp = client.embeddings.create(
        model="text-embedding-3-small",
        input=text,
    )
    return resp.data[0].embedding


def main():
    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "How does HRV relate to stress and meditation?"

    print(f"Collection: {COLLECTION}")
    print(f"Top K: {TOP_K}")
    print(f"Query: {query}")

    # Check collection info
    qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    info = qdrant.get_collection(COLLECTION)
    print(f"\nCollection info:")
    print(f"  Points: {info.points_count}")
    print(f"  Vectors config: {info.config.params.vectors}")
    print()

    # Generate query embedding
    print("Generating query embedding...")
    query_vec = get_embedding(query)
    print(f"  Embedding dim: {len(query_vec)}")

    # Search using query_points (qdrant-client >= 1.12)
    from qdrant_client.models import models

    print(f"\nSearching top {TOP_K} results...")
    response = qdrant.query_points(
        collection_name=COLLECTION,
        query=query_vec,
        limit=TOP_K,
        with_payload=True,
    )

    results = response.points if hasattr(response, "points") else response

    if not results:
        print("No results found!")
        return

    print(f"\n{'='*70}")
    for i, hit in enumerate(results, 1):
        payload = hit.payload or {}
        text = (payload.get("text") or payload.get("content") or "").strip()
        filename = payload.get("filename", "?")
        chunk_idx = payload.get("chunk_index", "?")
        score = hit.score

        preview = text[:200] + "..." if len(text) > 200 else text
        print(f"\n--- Result {i} (score: {score:.4f}) ---")
        print(f"  File: {filename}, chunk: {chunk_idx}")
        print(f"  Text: {preview}")

    print(f"\n{'='*70}")
    print(f"Total: {len(results)} results")


if __name__ == "__main__":
    main()
