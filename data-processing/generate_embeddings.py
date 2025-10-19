"""
Optimized script to generate embeddings for songs and store them in PostgreSQL.

This script efficiently processes songs by:
- Using async/concurrent API calls for parallel processing
- Batching database operations to minimize connection overhead
- Implementing robust retry logic with exponential backoff
- Supporting graceful interruption and progress tracking

When has_lyrics is true:
- If lyrics column is available: uses song_name, band, and lyrics for embeddings
- If lyrics column is unavailable: uses only song_name and band for embeddings

Usage:
    python generate_embeddings.py [OPTIONS]

Options:
    --limit=N           Process only N songs
    --table=NAME        Read from specific table (default: 'songs')
    --batch-size=N      Number of songs per API call (max 2048, default 500)
    --concurrency=N     Number of concurrent API calls (default: 5)
    --db-buffer=N       Number of batches to buffer before DB write (default: 5)

Examples:
    # Generate embeddings for 1000 songs
    python generate_embeddings.py --limit=1000

    # Process all songs with high concurrency
    python generate_embeddings.py --concurrency=10

    # Use larger batches with buffered writes
    python generate_embeddings.py --batch-size=1000 --db-buffer=10

Note: Press Ctrl+C to gracefully exit and save progress.
"""

import os
import sys
import time
import asyncio
from typing import Optional
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
from sqlalchemy import (
    create_engine,
    text,
    Table,
    Column,
    Integer,
    Float,
    MetaData,
)
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.exc import OperationalError, DatabaseError
from openai import OpenAI, OpenAIError

# Load environment variables
load_dotenv()

# Database configuration
POSTGRES_URL = os.environ.get("POSTGRES_URL")

# API Configuration
FIREWORKS_API_KEY = os.environ.get("FIREWORKS_API_KEY")
EMBEDDING_MODEL = "accounts/fireworks/models/qwen3-embedding-8b"
FIREWORKS_BASE_URL = "https://api.fireworks.ai/inference/v1"


def create_text_for_embedding(
    song_name: str, band: str, has_lyrics: bool, lyrics: Optional[str] = None
) -> str:
    """Create text for embedding based on available data."""
    if has_lyrics and lyrics:
        return f"{song_name} by {band}\n\n{lyrics}"
    return f"{song_name} by {band}"


def generate_embeddings_batch_with_retry(
    texts: list[str], max_retries: int = 3
) -> list[list[float]]:
    """
    Generate embeddings with exponential backoff retry logic.

    Args:
        texts: List of text strings to embed
        max_retries: Maximum number of retry attempts

    Returns:
        List of embedding vectors

    Raises:
        Exception: If all retries fail
    """
    client = OpenAI(base_url=FIREWORKS_BASE_URL, api_key=FIREWORKS_API_KEY)
    delay = 1

    for attempt in range(max_retries):
        try:
            response = client.embeddings.create(input=texts, model=EMBEDDING_MODEL)
            return [item.embedding for item in response.data]
        except OpenAIError as e:
            if attempt < max_retries - 1:
                print(
                    f"  API error (attempt {attempt + 1}/{max_retries}): {str(e)[:100]}"
                )
                print(f"  Retrying in {delay}s...")
                time.sleep(delay)
                delay *= 2
            else:
                raise
        except Exception as e:
            if attempt < max_retries - 1:
                print(
                    f"  Unexpected error (attempt {attempt + 1}/{max_retries}): {str(e)[:100]}"
                )
                print(f"  Retrying in {delay}s...")
                time.sleep(delay)
                delay *= 2
            else:
                raise


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
    )

    with engine.begin() as conn:
        embeddings_table.create(conn, checkfirst=True)

    return embeddings_table


def fetch_songs_to_process(
    engine, table_name: str, limit: Optional[int] = None
) -> list[dict]:
    """
    Fetch songs from database.

    Args:
        engine: SQLAlchemy engine
        table_name: Name of the songs table
        limit: Optional limit on number of songs to process

    Returns:
        List of song dictionaries

    Raises:
        ValueError: If table_name is not in the allowed whitelist
    """
    # Whitelist of allowed table names to prevent SQL injection
    allowed_tables = {"songs"}
    if table_name not in allowed_tables:
        raise ValueError(
            f"Invalid table name '{table_name}'. Allowed tables: {', '.join(sorted(allowed_tables))}"
        )

    with engine.connect() as conn:
        query = text(f"""
            SELECT song_id, song_name, band, has_lyrics, lyrics
            FROM {table_name}
            ORDER BY song_id
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


def store_embeddings_bulk(engine, embeddings_data: list[dict]):
    """
    Store embeddings in the database using bulk insert for better performance.
    Uses ON CONFLICT to update existing embeddings instead of creating duplicates.

    Args:
        engine: SQLAlchemy engine
        embeddings_data: List of dicts with song_id and embedding
    """
    if not embeddings_data:
        return

    with engine.begin() as conn:
        # Use executemany for better performance with large batches
        # ON CONFLICT clause prevents duplicates and updates existing embeddings
        insert_sql = text("""
            INSERT INTO song_embeddings (song_id, embedding)
            VALUES (:song_id, :embedding)
            ON CONFLICT (song_id) 
            DO UPDATE SET embedding = EXCLUDED.embedding
        """)
        conn.execute(insert_sql, embeddings_data)


async def process_batch_async(
    batch: list[dict], batch_num: int, total_batches: int, executor: ThreadPoolExecutor
) -> tuple[list[dict], bool]:
    """
    Process a single batch of songs asynchronously.

    Args:
        batch: List of song dictionaries
        batch_num: Current batch number
        total_batches: Total number of batches
        executor: ThreadPoolExecutor for running blocking API calls

    Returns:
        Tuple of (embeddings_data, success)
    """
    try:
        # Prepare texts for embedding
        texts = [
            create_text_for_embedding(
                song["song_name"], song["band"], song["has_lyrics"], song.get("lyrics")
            )
            for song in batch
        ]

        # Run API call in executor to avoid blocking
        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(
            executor, generate_embeddings_batch_with_retry, texts
        )

        # Prepare data for storage
        embeddings_data = [
            {
                "song_id": song["song_id"],
                "embedding": embedding,
            }
            for song, embedding in zip(batch, embeddings)
        ]

        return embeddings_data, True

    except Exception as e:
        print(f"  ❌ Error in batch {batch_num}/{total_batches}: {str(e)[:100]}")
        return [], False


async def process_batches_concurrently(
    songs: list[dict],
    batch_size: int,
    concurrency: int,
    db_buffer_size: int,
    engine,
    start_time: float,
) -> dict:
    """
    Process multiple batches concurrently with buffered database writes.

    Args:
        songs: List of all songs to process
        batch_size: Number of songs per API call
        concurrency: Number of concurrent API calls
        db_buffer_size: Number of batches to buffer before writing to DB
        engine: SQLAlchemy engine
        start_time: Start time for calculating elapsed time

    Returns:
        Statistics dictionary
    """
    stats = {
        "total_songs": len(songs),
        "embeddings_generated": 0,
        "embeddings_stored": 0,
        "errors": 0,
        "interrupted": False,
    }

    # Create batches
    batches = [songs[i : i + batch_size] for i in range(0, len(songs), batch_size)]
    total_batches = len(batches)

    # Buffer for database writes
    db_buffer = []

    # Track milestones for progress feedback
    last_milestone = 0
    milestone_interval = 10000

    # ThreadPoolExecutor for blocking API calls
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        try:
            # Process batches in chunks for concurrent execution
            for chunk_start in range(0, total_batches, concurrency):
                chunk_end = min(chunk_start + concurrency, total_batches)
                chunk_batches = batches[chunk_start:chunk_end]

                # Create tasks for concurrent processing
                tasks = [
                    process_batch_async(
                        batch, chunk_start + i + 1, total_batches, executor
                    )
                    for i, batch in enumerate(chunk_batches)
                ]

                print(
                    f"\nProcessing batches {chunk_start + 1}-{chunk_end}/{total_batches} concurrently..."
                )

                # Wait for all tasks in this chunk to complete
                results = await asyncio.gather(*tasks, return_exceptions=True)

                # Process results
                for i, result in enumerate(results):
                    if isinstance(result, Exception):
                        print(f"  ❌ Batch {chunk_start + i + 1} failed: {result}")
                        stats["errors"] += 1
                        continue

                    embeddings_data, success = result
                    if success and embeddings_data:
                        db_buffer.extend(embeddings_data)
                        stats["embeddings_generated"] += len(embeddings_data)

                        # Check for milestone feedback (every 10,000 embeddings)
                        current_count = stats["embeddings_generated"]
                        current_milestone = (
                            current_count // milestone_interval
                        ) * milestone_interval
                        if current_milestone > last_milestone and current_milestone > 0:
                            elapsed_time = time.time() - start_time
                            rate = (
                                current_count / elapsed_time if elapsed_time > 0 else 0
                            )
                            print(
                                f"\n  🎵 Milestone: {current_milestone:,} embeddings processed!"
                            )
                            print(f"     Rate: {rate:.2f} embeddings/second")
                            print(f"     Elapsed time: {elapsed_time:.1f}s\n")
                            last_milestone = current_milestone

                # Write to database when buffer is full or this is the last chunk
                if (
                    len(db_buffer) >= db_buffer_size * batch_size
                    or chunk_end == total_batches
                ):
                    if db_buffer:
                        print(
                            f"  💾 Writing {len(db_buffer)} embeddings to database..."
                        )
                        retry_with_backoff(
                            lambda: store_embeddings_bulk(engine, db_buffer)
                        )
                        stats["embeddings_stored"] += len(db_buffer)
                        db_buffer = []

                # Progress update
                songs_processed = min((chunk_end) * batch_size, len(songs))
                progress_pct = (songs_processed / len(songs)) * 100
                print(
                    f"  ✓ Progress: {songs_processed}/{len(songs)} ({progress_pct:.1f}%)"
                )

        except KeyboardInterrupt:
            print("\n\n⚠️  Interrupted by user. Saving buffered data...")
            if db_buffer:
                retry_with_backoff(lambda: store_embeddings_bulk(engine, db_buffer))
                stats["embeddings_stored"] += len(db_buffer)
            stats["interrupted"] = True

    return stats


def generate_and_store_embeddings(
    table_name: str = "songs",
    limit: Optional[int] = None,
    batch_size: int = 500,
    concurrency: int = 5,
    db_buffer_size: int = 5,
):
    """
    Main function to generate and store embeddings with optimized concurrent processing.

    Args:
        table_name: Name of the songs table
        limit: Optional limit on number of songs to process
        batch_size: Number of songs to process per API call
        concurrency: Number of concurrent API calls
        db_buffer_size: Number of batches to buffer before DB write

    Returns:
        Dictionary with statistics
    """
    start_time = time.time()

    print(f"\n{'=' * 80}")
    print("OPTIMIZED EMBEDDING GENERATION")
    print(f"{'=' * 80}")
    print(f"Source table: {table_name}")
    print(f"Batch size: {batch_size} songs per API call")
    print(f"Concurrency: {concurrency} parallel API calls")
    print(f"DB buffer: {db_buffer_size} batches")
    if limit:
        print(f"Processing up to {limit} songs...")
    else:
        print("Processing all songs...")
    print(f"{'=' * 80}\n")

    # Connect to database with optimized pool settings
    print("Connecting to PostgreSQL database...")
    engine = create_engine(
        POSTGRES_URL,
        pool_pre_ping=True,
        pool_size=10,  # Increased pool size for concurrent operations
        max_overflow=20,
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
            "total_time": 0,
        }

    # Process songs with async/concurrent execution
    stats = asyncio.run(
        process_batches_concurrently(
            songs, batch_size, concurrency, db_buffer_size, engine, start_time
        )
    )

    engine.dispose()

    # Calculate performance metrics
    total_time = time.time() - start_time
    stats["total_time"] = total_time
    if stats["embeddings_generated"] > 0:
        stats["songs_per_second"] = stats["embeddings_generated"] / total_time

    return stats


def main():
    """Main entry point for the script."""
    # Parse command line arguments
    limit = None
    table_name = "songs"
    batch_size = 500
    concurrency = 5
    db_buffer_size = 5

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
        elif arg.startswith("--concurrency="):
            try:
                concurrency = int(arg.split("=")[1])
                if concurrency < 1 or concurrency > 50:
                    print("Concurrency must be between 1 and 50")
                    sys.exit(1)
            except ValueError:
                print(f"Invalid concurrency value: {arg}")
                sys.exit(1)
        elif arg.startswith("--db-buffer="):
            try:
                db_buffer_size = int(arg.split("=")[1])
                if db_buffer_size < 1:
                    print("DB buffer size must be at least 1")
                    sys.exit(1)
            except ValueError:
                print(f"Invalid db-buffer value: {arg}")
                sys.exit(1)
        elif arg in ["--help", "-h"]:
            print(__doc__)
            sys.exit(0)

    try:
        # Run the optimized embedding generation pipeline
        stats = generate_and_store_embeddings(
            table_name=table_name,
            limit=limit,
            batch_size=batch_size,
            concurrency=concurrency,
            db_buffer_size=db_buffer_size,
        )

        # Print summary with performance metrics
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)
        print(f"Total songs: {stats['total_songs']}")
        print(f"Embeddings generated: {stats['embeddings_generated']}")
        print(f"Embeddings stored: {stats['embeddings_stored']}")
        print(f"Errors: {stats['errors']}")
        print(f"Total time: {stats.get('total_time', 0):.2f}s")
        if stats.get("songs_per_second"):
            print(f"Processing rate: {stats['songs_per_second']:.2f} songs/second")
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
