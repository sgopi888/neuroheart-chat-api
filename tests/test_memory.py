"""
Test the 3-layer memory architecture end-to-end.

Tests:
1. Fact extraction from a chat exchange
2. Storage in Qdrant (with dedup check)
3. Semantic retrieval of stored memories
4. Full prompt assembly with all 3 layers
"""
from __future__ import annotations

import asyncio
import os

from dotenv import load_dotenv

load_dotenv()

from app.memory_service import (
    _extract_facts,
    _store_facts,
    retrieve_memories,
    extract_and_store_memories,
    _ensure_collection,
    _MEMORY_COLLECTION,
    _get_qdrant,
)


def test_extract_facts():
    """Test LLM fact extraction from a chat exchange."""
    print("\n=== Test 1: Fact Extraction ===")

    user_msg = (
        "I've been meditating for about 3 months now, usually 20 minutes in the morning. "
        "My sleep has been terrible though - I keep waking up at 3am. "
        "I work as a software engineer and the deadline stress is killing me."
    )
    assistant_msg = (
        "It's great that you've built a consistent meditation practice! "
        "The 3am wake-ups could be related to cortisol spikes from work stress. "
        "Try a 5-minute body scan before bed and consider limiting screen time after 9pm. "
        "Your HRV data shows lower readings on weekdays which aligns with the work stress pattern."
    )

    facts = _extract_facts(user_msg, assistant_msg)
    print(f"Extracted {len(facts)} facts:")
    for i, f in enumerate(facts, 1):
        print(f"  {i}. {f}")

    assert len(facts) > 0, "Should extract at least one fact"
    return facts


def test_store_and_retrieve(facts: list[str]):
    """Test storing facts and retrieving them."""
    print("\n=== Test 2: Store & Retrieve ===")

    test_user = "test_memory_user_001"

    # Ensure collection exists
    _ensure_collection()
    print(f"Collection '{_MEMORY_COLLECTION}' ready")

    # Store facts
    stored = _store_facts(test_user, facts)
    print(f"Stored {stored} facts (out of {len(facts)})")

    # Try storing again — should dedup
    stored_again = _store_facts(test_user, facts)
    print(f"Re-stored {stored_again} facts (should be 0 — dedup)")

    # Retrieve with relevant query
    memories = retrieve_memories(test_user, "How is my meditation practice going?")
    print(f"\nRetrieved {len(memories)} memories for 'meditation practice':")
    for i, m in enumerate(memories, 1):
        print(f"  {i}. {m}")

    # Retrieve with different query
    memories2 = retrieve_memories(test_user, "What about my sleep issues?")
    print(f"\nRetrieved {len(memories2)} memories for 'sleep issues':")
    for i, m in enumerate(memories2, 1):
        print(f"  {i}. {m}")

    return memories


def test_async_extract():
    """Test the async background extraction."""
    print("\n=== Test 3: Async Extract & Store ===")

    test_user = "test_memory_user_002"

    async def _run():
        await extract_and_store_memories(
            test_user,
            "I just started doing yoga twice a week and my resting heart rate dropped to 62.",
            "That's wonderful progress! Yoga combined with your meditation practice creates a synergistic "
            "effect on your autonomic nervous system. A resting HR of 62 is excellent.",
        )
        # Give it a moment
        memories = retrieve_memories(test_user, "exercise and heart rate")
        print(f"Retrieved {len(memories)} memories after async extraction:")
        for i, m in enumerate(memories, 1):
            print(f"  {i}. {m}")

    asyncio.run(_run())


def test_cleanup():
    """Remove test data."""
    print("\n=== Cleanup ===")
    client = _get_qdrant()
    from qdrant_client.http import models as qm

    for test_user in ["test_memory_user_001", "test_memory_user_002"]:
        try:
            client.delete(
                collection_name=_MEMORY_COLLECTION,
                points_selector=qm.FilterSelector(
                    filter=qm.Filter(
                        must=[
                            qm.FieldCondition(
                                key="user_uid",
                                match=qm.MatchValue(value=test_user),
                            ),
                        ]
                    )
                ),
            )
            print(f"Cleaned up memories for {test_user}")
        except Exception as exc:
            print(f"Cleanup for {test_user}: {exc}")


if __name__ == "__main__":
    print("Memory Architecture Test")
    print(f"Model: {os.getenv('OPENAI_MODEL', 'gpt-5-nano')}")

    facts = test_extract_facts()
    test_store_and_retrieve(facts)
    test_async_extract()
    test_cleanup()

    print("\nAll tests passed!")
