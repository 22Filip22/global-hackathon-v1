import asyncio
from pathlib import Path
import shutil
import zipfile
import uuid
import random

from digesting import digest_directory
from vectorizer import add_chunks_to_qdrant
from ollama import chat
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance

# --- Configuration ---
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
fileCounter = 0

QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "knowledge"  # Use the same collection as in vectorizer.py
OLLAMA_MODEL = "gemma3:4b"  # Socratic chat model

# --- Qdrant client ---
client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

if not client.collection_exists(COLLECTION_NAME):
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=768, distance=Distance.COSINE)
    )

# --- Utilities ---
def get_random_node_from_db():
    """
    Pick a random node from Qdrant and return its payload.
    Raises an error if collection is empty.
    """
    all_points = []
    scroll_limit = 1000
    next_page = None

    while True:
        points, next_page = client.scroll(
            collection_name=COLLECTION_NAME,
            limit=scroll_limit,
            offset=0 if next_page is None else next_page
        )
        all_points.extend(points)
        if not next_page:  # reached the end
            break

    if not all_points:
        raise ValueError("Qdrant collection is empty. Add nodes first.")

    # pick a random point
    random_point = random.choice(all_points)

    # Make sure payload contains 'neighbors'
    neighbors = random_point.payload.get("neighbors", [])
    node = random_point.payload.get("node", "Unnamed Node")

    return node, neighbors

def socratic_question(node, neighbors):
    """Generate a Socratic question for the node using Ollama."""
    prompt = f"""
You are a teacher using the Socratic method.
Your goal is to ask guiding questions to help the learner understand the topic.

Current node: {node}
Neighbors: {neighbors}

Ask one guiding question to help the student explore this node and its neighbors.
"""
    response = chat(model=OLLAMA_MODEL, messages=[{"role": "user", "content": prompt}])
    return response.message.content

async def process_course(file_path: Path):
    """Upload a course zip file and add its chunks to Qdrant."""
    global fileCounter

    if not file_path.exists() or file_path.suffix != ".zip":
        print("Error: Only existing .zip files are allowed")
        return



    temp_path = UPLOAD_DIR / f"{fileCounter}.zip"
    shutil.copy(file_path, temp_path)

    target_dir = UPLOAD_DIR / str(fileCounter)
    target_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(temp_path, "r") as zip_ref:
        zip_ref.extractall(target_dir)
    temp_path.unlink()

    chunks = digest_directory(target_dir, chunk_pdf_by_page=True)
    print(f"Extracted {len(chunks)} chunks from uploaded course.")

    await add_chunks_to_qdrant(chunks)
    fileCounter += 1
    print(f"Upload complete. Upload ID: {fileCounter - 1}")

async def test_upload():
    """Upload dummy chunks for testing."""
    dummy_chunks = [
        "This is a test chunk about AI.",
        "Graphiti is a graph database for structured knowledge.",
        "FastAPI allows asynchronous endpoints easily."
    ]
    await add_chunks_to_qdrant(dummy_chunks)
    print(f"Dummy chunks uploaded. Total: {len(dummy_chunks)}")

def explore_graph():
    """Interactive Socratic graph exploration from Qdrant nodes."""
    try:
        current_node, neighbors = get_random_node_from_db()
    except ValueError:
        print("No nodes in Qdrant. Upload some courses first.")
        return

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
                current_node = next_node
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

# --- Interactive CLI ---
def main():
    while True:
        print("\n--- Graph Learning CLI ---")
        print("1. Upload course (.zip)")
        print("2. Test upload dummy chunks")
        print("3. Explore nodes interactively")
        print("4. Quit")

        choice = input("Select an option: ").strip()

        if choice == "1":
            path_str = input("Enter path to .zip course: ").strip().strip('"')
            asyncio.run(process_course(Path(path_str)))
        elif choice == "2":
            asyncio.run(test_upload())
        elif choice == "3":
            explore_graph()
        elif choice == "4":
            print("Goodbye!")
            break
        else:
            print("Invalid choice. Try again.")

if __name__ == "__main__":
    main()
