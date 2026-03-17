from qdrant_client import QdrantClient, models
from fastembed import SparseTextEmbedding
import uuid

# Connect
qdrant = QdrantClient(
    url="https://b110c5dc-dc97-4883-9598-d8c21e3e2f4b.us-east4-0.gcp.cloud.qdrant.io",
    api_key="YOUR_QDRANT_API_KEY",
)

# 1. Create new hybrid collection
qdrant.create_collection(
    collection_name="documents1_hybrid",
    vectors_config={
        "dense": models.VectorParams(size=1536, distance=models.Distance.COSINE),
    },
    sparse_vectors_config={
        "sparse": models.SparseVectorParams(),
    },
)

# 2. Init BM25 model
bm25_model = SparseTextEmbedding("Qdrant/bm25")

# 3. Scroll through all existing points and migrate
offset = None
batch_size = 100

while True:
    results, offset = qdrant.scroll(
        collection_name="documents1",
        limit=batch_size,
        offset=offset,
        with_vectors=True,
        with_payload=True,
    )

    if not results:
        break

    points = []
    for point in results:
        text = point.payload.get("text", "")
        dense_vector = point.vector  # existing OpenAI 1536-dim vector

        # Generate sparse BM25 vector from the text payload
        sparse_vec = list(bm25_model.query_embed(text))[0]

        points.append(
            models.PointStruct(
                id=point.id,
                vector={
                    "dense": dense_vector,
                    "sparse": models.SparseVector(
                        indices=sparse_vec.indices.tolist(),
                        values=sparse_vec.values.tolist(),
                    ),
                },
                payload=point.payload,
            )
        )

    qdrant.upsert(collection_name="documents1_hybrid", points=points)
    print(f"Migrated {len(points)} points")

    if offset is None:
        break

print("Done! Migration complete.")
# This preserves your original point IDs, payloads, and dense vectors — it just adds the sparse BM25 vector alongside. After verifying the new collection works, you can delete the old one or keep it as a backup.