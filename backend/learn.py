from ollama import chat
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance
import random

# --- Configuration ---
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "knowledge"  # Use the same collection as in vectorizer.py
OLLAMA_MODEL = "gemma3:4b"  # Chat model for Socratic questions

# --- Qdrant client ---
client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

# Ensure collection exists
if not client.collection_exists(COLLECTION_NAME):
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=768, distance=Distance.COSINE)
    )

# --- Utilities ---
def get_random_node_from_db():
    """Pick a random node from Qdrant and return its payload."""
    # scroll() returns a tuple (points, next_page)
    points, _ = client.scroll(collection_name=COLLECTION_NAME, limit=1000)
    if not points:
        raise ValueError("Qdrant collection is empty. Add nodes first.")
    random_point = random.choice(points)
    return random_point.payload["node"], random_point.payload["neighbors"]

def socratic_question(node, neighbors):
    """Ask Ollama to generate a Socratic question about the current node."""
    prompt = f"""
You are a teacher using the Socratic method.
Your goal is to ask guiding questions to help the learner understand the topic.

Current node: {node}
Neighbors: {neighbors}

Ask one guiding question to help the student explore this node and its neighbors.
"""
    response = chat(model=OLLAMA_MODEL, messages=[{"role": "user", "content": prompt}])
    return response.message.content

# --- Interactive exploration ---
def explore_graph():
    current_node, neighbors = get_random_node_from_db()

    while True:
        question = socratic_question(current_node, neighbors)
        print("\nTeacher (Socratic):", question)

        if neighbors:
            print("\nAvailable neighbors:", neighbors)
            next_node = input("Enter a neighbor to explore (or 'quit' to exit): ").strip()
            if next_node.lower() == "quit":
                print("Exiting exploration.")
                break
            elif next_node in neighbors:
                # Move to selected neighbor
                current_node = next_node
                # Fetch neighbors from Qdrant
                points, _ = client.scroll(collection_name=COLLECTION_NAME, limit=1000)
                for p in points:
                    if p.payload["node"] == current_node:
                        neighbors = p.payload["neighbors"]
                        break
            else:
                print("Invalid neighbor. Try again.")
        else:
            print("No neighbors to explore. Exploration ends here.")
            break

# --- Run ---
if __name__ == "__main__":
    explore_graph()
