"""
Semantic search operations for song retrieval.

This module handles:
- Database connection initialization
- pgvector similarity search
- Reranking by popularity score
"""

import os
import sys
from pathlib import Path
from typing import TypedDict
from sanic.log import logger
from sqlalchemy import text

# Add project root to path to import database module
sys.path.insert(0, str(Path(__file__).parent.parent))

from database.connection import get_engine, initialize_pgvector


class Song(TypedDict):
    """Type definition for a song."""

    title: str
    artist: str
    spotifyLink: str | None


# Initialize database connection with web server pooling
postgres_url = os.environ.get("POSTGRES_URL")
if postgres_url:
    db_engine = get_engine(for_web_server=True)
    logger.info("Database connection initialized with web server pooling")

    # Enable pgvector extension on startup
    if initialize_pgvector(for_web_server=True):
        logger.info("pgvector extension enabled")
    else:
        logger.error("Failed to enable pgvector extension")
else:
    logger.warning("POSTGRES_URL not found. Database queries will not work.")
    db_engine = None


async def search_database(
    query_embedding: list[float],
    n: int = 5,
) -> list[Song]:
    """
    Search the database for songs with similar embeddings using pgvector.

    Retrieves the most similar songs to the query embedding, then reranks them by popularity score.

    Args:
        query_embedding: The embedding vector to search for
        n: Number of results to return

    Returns:
        List of Song dictionaries with title and artist, ordered by popularity
    """
    if not db_engine:
        logger.error("Database not initialized. Cannot search for songs.")
        return []

    try:
        with db_engine.connect() as conn:
            # Convert embedding list to pgvector format (string representation)
            embedding_str = f"[{','.join(map(str, query_embedding))}]"

            query = text(
                """SELECT s.song_name, s.band, s.popularity_score
                FROM public.song_embeddings se
                JOIN public.songs s ON se.song_id = s.song_id
                ORDER BY se.embedding <=> :embedding 
                LIMIT :limit
                """
            )

            result = conn.execute(
                query,
                {
                    "embedding": embedding_str,
                    "limit": n,
                },
            )

            songs = []
            for row in result:
                songs.append(
                    {
                        "title": row[0],
                        "artist": row[1],
                        "popularity_score": row[2],
                        "spotifyLink": None,
                    }
                )

            # Rerank by popularity score (higher is better)
            songs.sort(key=lambda x: x["popularity_score"], reverse=True)

            # Remove popularity_score from final results
            for song in songs:
                del song["popularity_score"]

        return songs

    except Exception as e:
        logger.error(f"Database search error: {e}")
        # Return empty list on error rather than crashing
        return []
