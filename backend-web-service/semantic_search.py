"""
Semantic search operations for song retrieval.

This module handles:
- Database connection initialization
- pgvector similarity search
- Hybrid scoring (similarity + popularity)
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
    use_hybrid_scoring: bool = True,
    popularity_weight: float = 0.2,
) -> list[Song]:
    """
    Search the database for songs with similar embeddings using pgvector.

    Supports two scoring modes:
    1. Pure similarity: Orders by cosine distance only
    2. Hybrid scoring: Combines similarity with popularity metrics

    Hybrid scoring formula:
        hybrid_score = (1 - popularity_weight) × similarity_score + popularity_weight × normalized_popularity

    Where:
        - similarity_score = 1 - cosine_distance (higher is better)
        - normalized_popularity = popularity_score normalized to 0-1 range
        - popularity_weight controls the balance (0 = pure similarity, 1 = pure popularity)

    Args:
        query_embedding: The embedding vector to search for
        n: Number of results to return
        use_hybrid_scoring: If True, combines similarity with popularity
        popularity_weight: Weight for popularity (0.0 to 1.0). Default 0.2 means 80% similarity, 20% popularity

    Returns:
        List of Song dictionaries with title and artist
    """
    if not db_engine:
        logger.error("Database not initialized. Cannot search for songs.")
        return []

    try:
        with db_engine.connect() as conn:
            if use_hybrid_scoring:
                # Hybrid scoring: Get more candidates then re-rank with popularity
                # Fetch 3x the requested amount to ensure we have enough candidates after re-ranking
                candidate_limit = n * 3

                # Convert embedding list to pgvector format (string representation)
                embedding_str = f"[{','.join(map(str, query_embedding))}]"

                query = text("""
                    WITH ranked_embeddings AS (
                        SELECT 
                            e.song_id,
                            e.embedding
                        FROM song_embeddings e
                    ),
                    similarity_scores AS (
                        SELECT 
                            s.song_name, 
                            s.band,
                            CAST(re.embedding AS vector) <=> CAST(:embedding AS vector) AS distance,
                            s.popularity_score,
                            s.interactions_count,
                            s.unique_users
                        FROM ranked_embeddings re
                        JOIN songs s ON re.song_id = s.song_id
                        ORDER BY CAST(re.embedding AS vector) <=> CAST(:embedding AS vector)
                        LIMIT :candidate_limit
                    ),
                    normalized_scores AS (
                        SELECT 
                            song_name,
                            band,
                            distance,
                            popularity_score,
                            interactions_count,
                            unique_users,
                            -- Convert distance to similarity (lower distance = higher similarity)
                            (1 - distance) AS similarity,
                            -- Normalize popularity to 0-1 range using min-max scaling
                            CASE 
                                WHEN MAX(popularity_score) OVER () > MIN(popularity_score) OVER ()
                                THEN (popularity_score - MIN(popularity_score) OVER ()) / 
                                     (MAX(popularity_score) OVER () - MIN(popularity_score) OVER ())
                                ELSE 0.5
                            END AS normalized_popularity
                        FROM similarity_scores
                    )
                    SELECT 
                        song_name,
                        band,
                        -- Hybrid score: weighted combination of similarity and popularity
                        (1 - :popularity_weight) * similarity + :popularity_weight * normalized_popularity AS hybrid_score,
                        distance,
                        popularity_score,
                        interactions_count
                    FROM normalized_scores
                    ORDER BY hybrid_score DESC
                    LIMIT :limit
                """)

                result = conn.execute(
                    query,
                    {
                        "embedding": embedding_str,
                        "candidate_limit": candidate_limit,
                        "popularity_weight": popularity_weight,
                        "limit": n,
                    },
                )

                songs = []
                for row in result:
                    songs.append(
                        {
                            "title": row[0],
                            "artist": row[1],
                        }
                    )
                    logger.debug(
                        f"  {row[0]} by {row[1]} - "
                        f"hybrid_score: {row[2]:.4f}, "
                        f"distance: {row[3]:.4f}, "
                        f"popularity: {row[4]:.2f}, "
                        f"interactions: {row[5]}"
                    )

                logger.debug(
                    f"Found {len(songs)} songs using hybrid scoring (weight={popularity_weight})"
                )

            else:
                # Pure similarity search (original implementation)
                # Convert embedding list to pgvector format (string representation)
                embedding_str = f"[{','.join(map(str, query_embedding))}]"

                query = text("""
                    WITH ranked_embeddings AS (
                        SELECT 
                            e.song_id,
                            e.embedding
                        FROM song_embeddings e
                    )
                    SELECT s.song_name, s.band
                    FROM ranked_embeddings re
                    JOIN songs s ON re.song_id = s.song_id
                    ORDER BY CAST(re.embedding AS vector) <=> CAST(:embedding AS vector)
                    LIMIT :limit
                """)

                result = conn.execute(query, {"embedding": embedding_str, "limit": n})

                songs = []
                for row in result:
                    songs.append(
                        {
                            "title": row[0],
                            "artist": row[1],
                        }
                    )

                logger.debug(f"Found {len(songs)} songs using pure similarity search")

            return songs

    except Exception as e:
        logger.error(f"Database search error: {e}")
        # Return empty list on error rather than crashing
        return []
