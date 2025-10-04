# graph.py
import os
import asyncio
from datetime import datetime, timezone
from graphiti_core import Graphiti
from graphiti_core.nodes import EpisodeType
from graphiti_core.llm_client.config import LLMConfig
from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient
from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig
from graphiti_core.cross_encoder.openai_reranker_client import OpenAIRerankerClient

# -------------------------
# Neo4j config
# -------------------------
NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "testpassword")
print(f"[INFO] Neo4j config: URI={NEO4J_URI}, USER={NEO4J_USER}")

# -------------------------
# Ollama LLM config
# -------------------------
llm_config = LLMConfig(
    api_key="abc",             # Ollama doesn’t need a real key
    model="llama3.1:8b",
    small_model="gemma3:4b",
    base_url="http://localhost:11434/v1",
)
llm_client = OpenAIGenericClient(config=llm_config)
print("[INFO] Ollama LLM client initialized")

# -------------------------
# Embedding client
# -------------------------
embedder = OpenAIEmbedder(
    config=OpenAIEmbedderConfig(
        api_key="abc",
        embedding_model="nomic-embed-text",
        embedding_dim=768,
        base_url="http://localhost:11434/v1",
    )
)
print("[INFO] Embedding client initialized")

# -------------------------
# Cross-encoder for reranking
# -------------------------
cross_encoder = OpenAIRerankerClient(client=llm_client, config=llm_config)
print("[INFO] Cross-encoder client initialized")

# -------------------------
# Function to send chunks to Graphiti
# -------------------------
async def send_chunks_to_graph(chunks, source_description="Uploaded file"):
    print(f"[INFO] Sending {len(chunks)} chunks to Graphiti, source: {source_description}")
    graphiti = Graphiti(
        NEO4J_URI,
        NEO4J_USER,
        NEO4J_PASSWORD,
        llm_client=llm_client,
        embedder=embedder,
        cross_encoder=cross_encoder,
    )

    try:
        print("[INFO] Building indices and constraints in Neo4j...")
        await graphiti.build_indices_and_constraints()
        print("[INFO] Indices and constraints built successfully")

        async def add_chunk(i, chunk):
            await graphiti.add_episode(
                name=f"Chunk {i}",
                episode_body=chunk,
                source=EpisodeType.text,
                source_description=source_description,
                reference_time=datetime.now(timezone.utc),
            )
            print(f"[INFO] Chunk {i} added successfully")

        # Schedule all chunk additions concurrently
        await asyncio.gather(*(add_chunk(i, chunk) for i, chunk in enumerate(chunks)))

    finally:
        await graphiti.close()
        print("[INFO] Graphiti connection closed")


