"""
Script to generate embeddings for songs and store them in PostgreSQL.
This script reads songs from the database, generates embeddings based on available data,
and stores them in a separate song_embeddings table.

When has_lyrics is true:
- If lyrics column is available: uses song_name, band, and lyrics for embeddings
- If lyrics column is unavailable: uses only song_name and band for embeddings

Usage:
    python generate_embeddings.py [OPTIONS]

Options:
    --limit=N           Process only N songs
    --table=NAME        Read from specific table (default: 'songs')
    --batch-size=N      Number of songs per API call (max 2048, default 100)

Examples:
    # Generate embeddings for 100 songs
    python generate_embeddings.py --limit=100

    # Process all songs
    python generate_embeddings.py

    # Use specific table and smaller batch size
    python generate_embeddings.py --table=my_songs --batch-size=50

Note: Press Ctrl+C to gracefully exit after the current batch.
"""

import os
import sys
import time
from typing import Optional
from dotenv import load_dotenv
from sqlalchemy import (
    create_engine,
    text,
    Table,
    Column,
    Integer,
    String,
    Float,
    MetaData,
)
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.exc import OperationalError, DatabaseError
from openai import OpenAI

# Load environment variables
load_dotenv()

# Initialize OpenAI client for embeddings
client = OpenAI(
    base_url="https://api.fireworks.ai/inference/v1",
    api_key=os.environ.get("FIREWORKS_API_KEY"),
)

# Database configuration
POSTGRES_URL = os.environ.get(
    "POSTGRES_URL",
    "postgresql://ho1yshif:K2ytIVfh9qNu2Ig6ARnhIxWL6iRlHrnw@dpg-d3pj9lt6ubrc73f7fh20-a.oregon-postgres.render.com/render_take_home",
)


def create_text_for_embedding(
    song_name: str, band: str, has_lyrics: bool, lyrics: Optional[str] = None
) -> str:
    """
    Create text for embedding based on available data.

    Args:
        song_name: Name of the song
        band: Band/artist name
        has_lyrics: Whether the song has lyrics
        lyrics: Optional lyrics text

    Returns:
        Formatted text for embedding generation
    """
    if has_lyrics and lyrics:
        # Use song_name, band, and lyrics
        return f"{song_name} by {band}\n\n{lyrics}"
    else:
        # Use only song_name and band
        return f"{song_name} by {band}"


def generate_embeddings_batch(texts: list[str]) -> list[list[float]]:
    """
    Generate embeddings for a batch of texts using Fireworks API.

    Args:
        texts: List of text strings to embed

    Returns:
        List of embedding vectors
    """
    response = client.embeddings.create(
        input=texts,
        model="accounts/fireworks/models/qwen3-embedding-8b",
    )
    return [item.embedding for item in response.data]


def retry_with_backoff(func, max_retries=5, initial_delay=1):
    """Retry a function with exponential backoff."""
    delay = initial_delay
    last_exception = None

    for attempt in range(max_retries):
        try:
            return func()
        except (OperationalError, DatabaseError) as e:
            last_exception = e
            if attempt < max_retries - 1:
                print(
                    f"  Connection error (attempt {attempt + 1}/{max_retries}): {str(e)[:100]}"
                )
                print(f"  Retrying in {delay} seconds...")
                time.sleep(delay)
                delay *= 2
            else:
                print(f"  Failed after {max_retries} attempts")
                raise last_exception


def setup_embeddings_table(engine):
    """Create song_embeddings table if it doesn't exist."""
    metadata = MetaData()
    embeddings_table = Table(
        "song_embeddings",
        metadata,
        Column("song_id", Integer, primary_key=True),
        Column("embedding", ARRAY(Float)),
        Column("text_used", String),  # Store the text that was used for embedding
    )

    with engine.begin() as conn:
        embeddings_table.create(conn, checkfirst=True)

    return embeddings_table


def fetch_songs_to_process(
    engine, table_name: str, limit: Optional[int] = None
) -> list[dict]:
    """
    Fetch songs from database that need embeddings.

    Args:
        engine: SQLAlchemy engine
        table_name: Name of the songs table
        limit: Optional limit on number of songs to process

    Returns:
        List of song dictionaries
    """
    with engine.connect() as conn:
        # Get songs that don't have embeddings yet
        query = text(f"""
            SELECT s.song_id, s.song_name, s.band, s.has_lyrics, s.lyrics
            FROM {table_name} s
            LEFT JOIN song_embeddings e ON s.song_id = e.song_id
            WHERE e.song_id IS NULL
            ORDER BY s.song_id
            {f"LIMIT {limit}" if limit else ""}
        """)
        result = conn.execute(query)
        songs = []
        for row in result:
            songs.append(
                {
                    "song_id": row[0],
                    "song_name": row[1],
                    "band": row[2],
                    "has_lyrics": row[3] if len(row) > 3 else False,
                    "lyrics": row[4] if len(row) > 4 else None,
                }
            )
        return songs


def store_embeddings(engine, embeddings_data: list[dict]):
    """
    Store embeddings in the database.

    Args:
        engine: SQLAlchemy engine
        embeddings_data: List of dicts with song_id, embedding, and text_used
    """
    with engine.begin() as conn:
        insert_sql = text("""
            INSERT INTO song_embeddings (song_id, embedding, text_used)
            VALUES (:song_id, :embedding, :text_used)
            ON CONFLICT (song_id) 
            DO UPDATE SET 
                embedding = EXCLUDED.embedding,
                text_used = EXCLUDED.text_used
        """)
        conn.execute(insert_sql, embeddings_data)


def generate_and_store_embeddings(
    table_name: str = "songs",
    limit: Optional[int] = None,
    batch_size: int = 100,
):
    """
    Main function to generate and store embeddings.

    Args:
        table_name: Name of the songs table
        limit: Optional limit on number of songs to process
        batch_size: Number of songs to process per API call

    Returns:
        Dictionary with statistics
    """
    print(f"\n{'=' * 80}")
    print("EMBEDDING GENERATION")
    print(f"{'=' * 80}")
    print(f"Source table: {table_name}")
    print(f"Batch size: {batch_size} songs per API call")
    if limit:
        print(f"Processing up to {limit} songs...")
    else:
        print("Processing all songs without embeddings...")
    print(f"{'=' * 80}\n")

    # Connect to database
    print("Connecting to PostgreSQL database...")
    engine = create_engine(
        POSTGRES_URL,
        pool_pre_ping=True,
        pool_recycle=3600,
        connect_args={
            "connect_timeout": 10,
            "keepalives": 1,
            "keepalives_idle": 30,
            "keepalives_interval": 10,
            "keepalives_count": 5,
        },
    )

    # Setup embeddings table
    print("Setting up song_embeddings table...")
    retry_with_backoff(lambda: setup_embeddings_table(engine))
    print("✓ Table setup complete\n")

    # Fetch songs
    print("Fetching songs to process...")
    songs = retry_with_backoff(
        lambda: fetch_songs_to_process(engine, table_name, limit)
    )
    print(f"Found {len(songs)} songs to process\n")

    if not songs:
        print("No songs to process. Exiting.")
        engine.dispose()
        return {
            "total_songs": 0,
            "embeddings_generated": 0,
            "embeddings_stored": 0,
            "errors": 0,
        }

    # Process songs in batches
    stats = {
        "total_songs": len(songs),
        "embeddings_generated": 0,
        "embeddings_stored": 0,
        "errors": 0,
        "interrupted": False,
    }

    try:
        for i in range(0, len(songs), batch_size):
            batch = songs[i : i + batch_size]
            batch_num = (i // batch_size) + 1
            total_batches = (len(songs) + batch_size - 1) // batch_size

            print(
                f"Processing batch {batch_num}/{total_batches} ({len(batch)} songs)..."
            )

            # Prepare texts for embedding
            texts = []
            for song in batch:
                text = create_text_for_embedding(
                    song["song_name"],
                    song["band"],
                    song["has_lyrics"],
                    song.get("lyrics"),
                )
                texts.append(text)

            # Generate embeddings
            try:
                print("  Generating embeddings...")
                embeddings = generate_embeddings_batch(texts)
                stats["embeddings_generated"] += len(embeddings)

                # Prepare data for storage
                embeddings_data = []
                for song, embedding, text in zip(batch, embeddings, texts):
                    embeddings_data.append(
                        {
                            "song_id": song["song_id"],
                            "embedding": embedding,
                            "text_used": text[:500] + "..."
                            if len(text) > 500
                            else text,  # Truncate for storage
                        }
                    )

                # Store embeddings
                print("  Storing embeddings in database...")
                retry_with_backoff(lambda: store_embeddings(engine, embeddings_data))
                stats["embeddings_stored"] += len(embeddings_data)

                progress_pct = ((i + len(batch)) / len(songs)) * 100
                print(
                    f"  ✓ Batch complete. Progress: {i + len(batch)}/{len(songs)} ({progress_pct:.1f}%)\n"
                )

            except Exception as e:
                print(f"  ❌ Error processing batch: {e}")
                stats["errors"] += 1
                continue

    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user. Saving progress...")
        stats["interrupted"] = True

    engine.dispose()
    return stats


def main():
    """Main entry point for the script."""
    # Parse command line arguments
    limit = None
    table_name = "songs"
    batch_size = 100

    for arg in sys.argv[1:]:
        if arg.startswith("--limit="):
            try:
                limit = int(arg.split("=")[1])
            except ValueError:
                print(f"Invalid limit value: {arg}")
                sys.exit(1)
        elif arg.startswith("--table="):
            table_name = arg.split("=")[1]
        elif arg.startswith("--batch-size="):
            try:
                batch_size = int(arg.split("=")[1])
                if batch_size < 1 or batch_size > 2048:
                    print("Batch size must be between 1 and 2048")
                    sys.exit(1)
            except ValueError:
                print(f"Invalid batch size value: {arg}")
                sys.exit(1)
        elif arg in ["--help", "-h"]:
            print(__doc__)
            sys.exit(0)

    try:
        # Run the embedding generation pipeline
        stats = generate_and_store_embeddings(
            table_name=table_name,
            limit=limit,
            batch_size=batch_size,
        )

        # Print summary
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)
        print(f"Total songs: {stats['total_songs']}")
        print(f"Embeddings generated: {stats['embeddings_generated']}")
        print(f"Embeddings stored: {stats['embeddings_stored']}")
        print(f"Errors: {stats['errors']}")
        if stats.get("interrupted"):
            print("\n⚠️  Process was interrupted")
            print("You can re-run this script to continue from where you left off.")
        print("=" * 80)

        # Exit with error code if interrupted
        if stats.get("interrupted"):
            sys.exit(130)

    except KeyboardInterrupt:
        print("\n\n⚠️  Force quit detected. Exiting immediately.")
        sys.exit(130)
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
