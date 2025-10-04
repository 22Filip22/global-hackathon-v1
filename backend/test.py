
import asyncio
import json
import logging
import os
from datetime import datetime, timezone
from logging import INFO

from dotenv import load_dotenv

from graphiti_core import Graphiti
from graphiti_core.cross_encoder import OpenAIRerankerClient
from graphiti_core.embedder import OpenAIEmbedder, OpenAIEmbedderConfig

from graphiti_core.llm_client import LLMConfig
from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient


from graphiti_core.nodes import EpisodeType

#################################################
# CONFIGURATION
#################################################
# Set up logging and environment variables for
# connecting to Neo4j database
#################################################

# Configure logging
logging.basicConfig(
    level=INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger(__name__)

load_dotenv()

# Neo4j connection parameters
# Make sure Neo4j Desktop is running with a local DBMS started


NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "testpassword")


async def main():
    #################################################
    # INITIALIZATION
    #################################################
    # Connect to Neo4j and set up Graphiti indices
    # This is required before using other Graphiti
    # functionality
    #################################################

    llm_config = LLMConfig(
        api_key="ollama",  # Ollama doesn't require a real API key, but some placeholder is needed
        model="deepseek-r1:7b",
        small_model="deepseek-r1:7b",
        base_url="http://127.0.0.1:11434/v1",  # Ollama's OpenAI-compatible endpoint
    )

    llm_client = OpenAIGenericClient(config=llm_config)

    graphiti = Graphiti(
        NEO4J_URI,
        NEO4J_USER,
        NEO4J_PASSWORD,
        llm_client=llm_client,
        embedder=OpenAIEmbedder(
            config=OpenAIEmbedderConfig(
                api_key="ollama",
                embedding_model="nomic-embed-text",
                embedding_dim=768,
                base_url="http://127.0.0.1:11434/v1",
            )
        ),
        cross_encoder=OpenAIRerankerClient(client=llm_client, config=llm_config),
    )


    try:
        # Initialize the graph database with graphiti's indices. This only needs to be done once.
        await graphiti.build_indices_and_constraints()

        #################################################
        # ADDING EPISODES
        #################################################
        # Episodes are the primary units of information
        # in Graphiti. They can be text or structured JSON
        # and are automatically processed to extract entities
        # and relationships.
        #################################################

        # Example: Add Episodes
        # Episodes list containing both text and JSON episodes
        episodes = [
            {
                'content': 'Claude is the flagship AI assistant from Anthropic. It was previously '
                'known as Claude Instant in its earlier versions.',
                'type': EpisodeType.text,
                'description': 'AI podcast transcript',
            },
            {
                'content': 'As an AI assistant, Claude has been available since December 15, 2022 – Present',
                'type': EpisodeType.text,
                'description': 'AI podcast transcript',
            },
            {
                'content': {
                    'name': 'GPT-4',
                    'creator': 'OpenAI',
                    'capability': 'Multimodal Reasoning',
                    'previous_version': 'GPT-3.5',
                    'training_data_cutoff': 'April 2023',
                },
                'type': EpisodeType.json,
                'description': 'AI model metadata',
            },
            {
                'content': {
                    'name': 'GPT-4',
                    'release_date': 'March 14, 2023',
                    'context_window': '128,000 tokens',
                    'status': 'Active',
                },
                'type': EpisodeType.json,
                'description': 'AI model metadata',
            },
        ]

        # Add episodes to the graph
        for i, episode in enumerate(episodes):
            await graphiti.add_episode(
                name=f'AI Agents Unleashed {i}',
                episode_body=episode['content']
                if isinstance(episode['content'], str)
                else json.dumps(episode['content']),
                source=episode['type'],
                source_description=episode['description'],
                reference_time=datetime.now(timezone.utc),
            )
            print(f'Added episode: AI Agents Unleashed {i} ({episode["type"].value})')

        #################################################
        # BASIC SEARCH
        #################################################
        # The simplest way to retrieve relationships (edges)
        # from Graphiti is using the search method, which
        # performs a hybrid search combining semantic
        # similarity and BM25 text retrieval.
        #################################################



    finally:
        #################################################
        # CLEANUP
        #################################################
        # Always close the connection to Neo4j when
        # finished to properly release resources
        #################################################

        # Close the connection
        await graphiti.close()
        print('\nConnection closed')


if __name__ == '__main__':
    asyncio.run(main())