from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance
from ollama import embed
import uuid

# --- Configuration ---
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "knowledge"  # use a new collection
OLLAMA_MODEL = "nomic-embed-text:latest"
VECTOR_SIZE = 768  # dimension for nomic-embed-text

# --- Initialize Qdrant client ---
client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

# Delete old collection if it exists, then create new one
if client.collection_exists(collection_name=COLLECTION_NAME):
    client.delete_collection(collection_name=COLLECTION_NAME)

client.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE)
)

def embed_chunks(chunks):
    """Generate embeddings for multiple chunks at once using Ollama."""
    response = embed(model=OLLAMA_MODEL, input=chunks)
    if not response.embeddings:
        raise ValueError("No embeddings returned from Ollama")
    return response.embeddings

async def add_chunks_to_qdrant(chunks):
    """Add a list of text chunks to Qdrant."""
    vectors = embed_chunks(chunks)
    points = []

    for chunk, vector in zip(chunks, vectors):
        points.append({
            "id": str(uuid.uuid4()),
            "vector": vector,
            "payload": {"text": chunk}
        })

    client.upsert(
        collection_name=COLLECTION_NAME,
        points=points
    )
    print(f"Inserted {len(points)} chunks into Qdrant.")

# --- Example usage ---
if __name__ == "__main__":
    text_chunks = [
        "Hello, this is the first chunk.",
        "This is another chunk of text.",
        "Qdrant stores embeddings for semantic search.",
        "Ollama can generate embeddings locally."
    ]
    add_chunks_to_qdrant(text_chunks)
