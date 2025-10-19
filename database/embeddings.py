"""
Database operations for embeddings.
"""

from typing import Optional, Dict, Any
import signal
import time
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import polars as pl
from sqlalchemy import text
from database.connection import get_connection
from database.embedding_client import EmbeddingClient


# Global flag for interrupt handling
_interrupted = False


def signal_handler(signum, frame):
    """Handle keyboard interrupt (Ctrl+C)."""
    global _interrupted
    _interrupted = True
    print(
        "\n\n⚠️  Keyboard interrupt received. Finishing current batch and exiting gracefully..."
    )
    print("⚠️  Press Ctrl+C again to force quit (may lose progress)")


class RateLimiter:
    """Thread-safe rate limiter using token bucket algorithm."""

    def __init__(self, requests_per_minute=6000):
        self.requests_per_minute = requests_per_minute
        self.tokens = requests_per_minute
        self.max_tokens = requests_per_minute
        self.last_update = time.time()
        self.lock = Lock()

    def acquire(self):
        """Wait until a token is available, then consume it."""
        while True:
            with self.lock:
                now = time.time()
                elapsed = now - self.last_update

                # Add tokens based on time elapsed
                self.tokens = min(
                    self.max_tokens,
                    self.tokens + (elapsed * self.requests_per_minute / 60),
                )
                self.last_update = now

                if self.tokens >= 1:
                    self.tokens -= 1
                    return

            # Wait a bit before trying again
            time.sleep(0.01)


def process_batch_with_retry(
    batch_num, batch_df, embedding_client, rate_limiter, max_retries=5
):
    """Process a single batch with exponential backoff on rate limits."""

    # Prepare texts for embedding
    texts_for_embedding = []
    batch_embedding_types = []

    for row in batch_df.iter_rows(named=True):
        lyrics = row.get("lyrics", "")
        if lyrics and str(lyrics).strip():
            text = lyrics
            embedding_type = "lyrics"
        else:
            song_name = row.get("song_name", "")
            band = row.get("band", "")
            text = f"{song_name} by {band}" if song_name and band else song_name or band
            embedding_type = "metadata"

        texts_for_embedding.append(text if text else "")
        batch_embedding_types.append(embedding_type)

    # Retry logic with exponential backoff
    for attempt in range(max_retries):
        try:
            # Wait for rate limiter
            rate_limiter.acquire()

            # Generate embeddings
            batch_embeddings = embedding_client.generate_embeddings_batch(
                texts_for_embedding, retry_on_rate_limit=True
            )

            return {
                "batch_num": batch_num,
                "embeddings": batch_embeddings,
                "types": batch_embedding_types,
                "errors": sum(1 for e in batch_embeddings if e is None),
                "success": True,
            }

        except Exception as e:
            error_str = str(e).lower()
            if "rate limit" in error_str or "429" in error_str:
                # Exponential backoff for rate limits
                wait_time = (2**attempt) + random.uniform(0, 1)
                print(
                    f"  Batch {batch_num + 1}: Rate limit hit, retrying in {wait_time:.1f}s..."
                )
                time.sleep(wait_time)
            else:
                # Non-rate-limit error, fail after exponential backoff
                if attempt < max_retries - 1:
                    wait_time = (2**attempt) + random.uniform(0, 1)
                    print(
                        f"  Batch {batch_num + 1}: Error ({e}), retrying in {wait_time:.1f}s..."
                    )
                    time.sleep(wait_time)
                else:
                    print(
                        f"  Batch {batch_num + 1}: Failed after {max_retries} attempts: {e}"
                    )
                    return {
                        "batch_num": batch_num,
                        "embeddings": [None] * len(texts_for_embedding),
                        "types": batch_embedding_types,
                        "errors": len(texts_for_embedding),
                        "success": False,
                    }

    # Max retries exceeded
    return {
        "batch_num": batch_num,
        "embeddings": [None] * len(texts_for_embedding),
        "types": batch_embedding_types,
        "errors": len(texts_for_embedding),
        "success": False,
    }


def read_songs_from_postgres(
    table_name: str = None,
    limit: Optional[int] = None,
    postgres_url: str = None,
) -> pl.DataFrame:
    """
    Read songs data from PostgreSQL database.

    Args:
        table_name: Name of the table to read from (defaults to songs table)
        limit: Optional limit on number of rows to read
        postgres_url: PostgreSQL connection URL (defaults to POSTGRES_URL env var)

    Returns:
        Polars DataFrame containing songs data with lyrics column (will use lyrics if populated, otherwise metadata)
    """
    # Always default to table with lyrics
    if table_name is None:
        table_name = "songs"

    # Always include lyrics column (only if populated)
    selected_columns = ["song_id", "song_name", "band", "lyrics"]
    columns_str = ", ".join(selected_columns)

    print(f"Reading songs from table '{table_name}'...")
    print("Mode: Will use lyrics if populated, otherwise metadata (song_name + band)")
    print(f"Selecting columns: {columns_str}")

    query = f"SELECT {columns_str} FROM {table_name}"
    if limit:
        query += f" LIMIT {limit}"

    with get_connection(postgres_url) as conn:
        result = conn.execute(text(query))
        rows = result.fetchall()
        columns = result.keys()

    # Convert to Polars DataFrame with progress tracking for every 10K rows
    print("Processing rows from database...")
    df_data = {}
    for i, col in enumerate(columns):
        df_data[col] = []

    for row_idx, row in enumerate(rows):
        for i, col in enumerate(columns):
            df_data[col].append(row[i])

        # Progress tracking for every 10K values read
        if (row_idx + 1) % 10000 == 0:
            print(f"  Read progress: {row_idx + 1:,} rows processed...")

    df = pl.DataFrame(df_data)
    print(f"✓ Read {len(df):,} songs from database")

    return df


def check_existing_embeddings(
    df: pl.DataFrame,
    embedding_type: str,
    postgres_url: str = None,
) -> pl.DataFrame:
    """
    Check which songs already have embeddings and filter them out.

    Args:
        df: DataFrame containing songs with 'song_id' column
        embedding_type: Type of embedding ('metadata' or 'lyrics')
        postgres_url: PostgreSQL connection URL (defaults to POSTGRES_URL env var)

    Returns:
        DataFrame filtered to only songs that don't have embeddings yet
    """
    print(f"\nChecking for existing {embedding_type} embeddings...")

    with get_connection(postgres_url) as conn:
        result = conn.execute(
            text(f"""
                SELECT song_id 
                FROM {"song_embeddings"}
                WHERE embedding_type = :embedding_type
            """),
            {"embedding_type": embedding_type},
        )
        existing_song_ids = {row[0] for row in result.fetchall()}

    if existing_song_ids:
        print(f"  Found {len(existing_song_ids):,} songs with existing embeddings")
        df_filtered = df.filter(~pl.col("song_id").is_in(list(existing_song_ids)))
        print(f"  Will process {len(df_filtered):,} new songs")
        return df_filtered
    else:
        print(f"  No existing embeddings found, will process all {len(df):,} songs")
        return df


def store_embeddings(
    df: pl.DataFrame,
    embedding_type: str,
    postgres_url: str = None,
    batch_size: int = 100,
) -> int:
    """
    Store embeddings in the song_embeddings table.

    Args:
        df: DataFrame containing songs with embeddings (must have 'song_id' and 'embedding' columns)
        embedding_type: Type of embedding ('metadata' or 'lyrics')
        postgres_url: PostgreSQL connection URL (defaults to POSTGRES_URL env var)
        batch_size: Number of embeddings to insert per batch

    Returns:
        Number of embeddings stored
    """
    print(f"\nStoring {embedding_type} embeddings in '{'song_embeddings'}' table...")

    # Filter out rows without embeddings
    df_with_embeddings = df.filter(pl.col("embedding").is_not_null())

    print(f"Inserting {len(df_with_embeddings):,} embeddings...")

    stored_count = 0
    with get_connection(postgres_url) as conn:
        with conn.begin():
            for i, row in enumerate(df_with_embeddings.iter_rows(named=True)):
                if _interrupted:
                    print(
                        f"\n⚠️  Interrupted. Stored {stored_count:,} embeddings before interruption."
                    )
                    break

                song_id = row["song_id"]
                embedding = row["embedding"]

                # Convert embedding list to pgvector format
                embedding_str = "[" + ",".join(map(str, embedding)) + "]"

                # Use INSERT ... ON CONFLICT to handle duplicates
                conn.execute(
                    text(f"""
                        INSERT INTO {"song_embeddings"} 
                        (song_id, {"embedding"}, embedding_type, model_name)
                        VALUES (:song_id, :embedding::vector, :embedding_type, :model_name)
                        ON CONFLICT (song_id, embedding_type) 
                        DO UPDATE SET 
                            {"embedding"} = EXCLUDED.{"embedding"},
                            model_name = EXCLUDED.model_name,
                            created_at = NOW()
                    """),
                    {
                        "song_id": song_id,
                        "embedding": embedding_str,
                        "embedding_type": embedding_type,
                        "model_name": "qwen3-embedding-8b",
                    },
                )
                stored_count += 1

                if (i + 1) % batch_size == 0:
                    print(
                        f"  Progress: {i + 1:,}/{len(df_with_embeddings):,} embeddings stored..."
                    )

    if not _interrupted:
        # Verify the data was stored
        with get_connection(postgres_url) as conn:
            result = conn.execute(
                text(f"""
                    SELECT COUNT(*) 
                    FROM {"song_embeddings"}
                    WHERE embedding_type = :embedding_type
                """),
                {"embedding_type": embedding_type},
            )
            count = result.fetchone()[0]
            print(
                f"✓ Successfully stored {stored_count:,} new embeddings ({count:,} total in database)"
            )

    return stored_count


def store_embeddings_with_types(
    df: pl.DataFrame,
    postgres_url: str = None,
    batch_size: int = 1000,
) -> int:
    """
    Store embeddings in the song_embeddings table with their respective embedding types.
    Uses bulk inserts for better performance.

    Args:
        df: DataFrame containing songs with embeddings (must have 'song_id', 'embedding', and 'embedding_type' columns)
        postgres_url: PostgreSQL connection URL (defaults to POSTGRES_URL env var)
        batch_size: Number of embeddings to insert per batch (default 1000)

    Returns:
        Number of embeddings stored
    """
    print(f"\nStoring embeddings with dynamic types in '{'song_embeddings'}' table...")

    # Filter out rows without embeddings
    df_with_embeddings = df.filter(pl.col("embedding").is_not_null())

    print(
        f"Inserting {len(df_with_embeddings):,} embeddings in batches of {batch_size}..."
    )

    stored_count = 0
    total_batches = (len(df_with_embeddings) + batch_size - 1) // batch_size

    with get_connection(postgres_url) as conn:
        for batch_num in range(total_batches):
            if _interrupted:
                print(
                    f"\n⚠️  Interrupted. Stored {stored_count:,} embeddings before interruption."
                )
                break

            start_idx = batch_num * batch_size
            end_idx = min((batch_num + 1) * batch_size, len(df_with_embeddings))
            batch_df = df_with_embeddings.slice(start_idx, end_idx - start_idx)

            # Build bulk insert values
            values_list = []
            for row in batch_df.iter_rows(named=True):
                song_id = row["song_id"]
                embedding = row["embedding"]
                embedding_type = row["embedding_type"]
                embedding_str = "[" + ",".join(map(str, embedding)) + "]"

                # Escape single quotes in embedding_type for SQL safety
                embedding_type_safe = embedding_type.replace("'", "''")

                values_list.append(
                    f"({song_id}, '{embedding_str}'::vector, '{embedding_type_safe}', 'qwen3-embedding-8b')"
                )

            values_str = ",\n".join(values_list)

            # Execute bulk insert
            with conn.begin():
                conn.execute(
                    text(f"""
                        INSERT INTO song_embeddings 
                        (song_id, embedding, embedding_type, model_name)
                        VALUES {values_str}
                        ON CONFLICT (song_id, embedding_type) 
                        DO UPDATE SET 
                            embedding = EXCLUDED.embedding,
                            model_name = EXCLUDED.model_name,
                            created_at = NOW()
                    """)
                )

            stored_count += len(batch_df)

            if (batch_num + 1) % 10 == 0 or batch_num + 1 == total_batches:
                print(
                    f"  Progress: {stored_count:,}/{len(df_with_embeddings):,} embeddings stored..."
                )

    if not _interrupted:
        # Verify the data was stored
        with get_connection(postgres_url) as conn:
            result = conn.execute(
                text(f"""
                    SELECT embedding_type, COUNT(*) 
                    FROM {"song_embeddings"}
                    GROUP BY embedding_type
                """)
            )
            counts = result.fetchall()
            print(f"✓ Successfully stored {stored_count:,} new embeddings")
            print("  Total counts by type:")
            for embedding_type, count in counts:
                print(f"    {embedding_type}: {count:,}")

    return stored_count


def generate_and_store_all_embeddings(
    table_name: str = None,
    limit: Optional[int] = None,
    postgres_url: str = None,
    skip_existing: bool = True,
    batch_size: int = 2048,
) -> Dict[str, Any]:
    """
    Complete pipeline: read songs, generate embeddings, store in database.

    Uses batch processing to send up to 2048 songs per API call for efficiency.
    According to Fireworks AI docs: https://docs.fireworks.ai/api-reference/creates-an-embedding-vector-representing-the-input-text
    - Can send up to 2048 strings per request
    - Rate limit: 6000 requests per minute

    Automatically uses lyrics if the lyrics column is populated, otherwise falls back to metadata (song_name + band).

    Args:
        table_name: Name of the table to process (defaults to songs)
        limit: Optional limit on number of songs to process
        postgres_url: PostgreSQL connection URL (defaults to POSTGRES_URL env var)
        skip_existing: If True, skip songs that already have embeddings
        batch_size: Number of songs to process per API call (max 2048, default 2048)

    Returns:
        Dictionary with statistics about the operation
    """
    # Register signal handler for Ctrl+C
    global _interrupted
    _interrupted = False
    signal.signal(signal.SIGINT, signal_handler)

    # Validate batch size
    if batch_size > 2048:
        print(
            f"⚠️  Warning: batch_size {batch_size} exceeds API limit of 2048. Using 2048."
        )
        batch_size = 2048

    print("=" * 80)
    print("EMBEDDING GENERATION PIPELINE (BATCH MODE)")
    print("=" * 80)
    print("Mode: Will use lyrics if populated, otherwise metadata")
    print(f"Model: {'qwen3-embedding-8b'}")
    print(f"Batch size: {batch_size} songs per API call")

    try:
        # Step 1: Read songs from database
        df = read_songs_from_postgres(table_name, limit, postgres_url)

        # Add a column to track which embedding type to use for each song
        # Use lyrics if populated, otherwise metadata
        def determine_embedding_type(lyrics):
            return "lyrics" if lyrics and str(lyrics).strip() else "metadata"

        df = df.with_columns(
            pl.col("lyrics")
            .map_elements(determine_embedding_type, return_dtype=pl.Utf8)
            .alias("embedding_type_to_use")
        )

        # Count how many will use lyrics vs metadata
        lyrics_count = df.filter(pl.col("embedding_type_to_use") == "lyrics").shape[0]
        metadata_count = df.filter(pl.col("embedding_type_to_use") == "metadata").shape[
            0
        ]
        print(f"\nSongs with lyrics: {lyrics_count:,}")
        print(f"Songs without lyrics (will use metadata): {metadata_count:,}")

        # Note: For simplicity, we'll skip checking existing embeddings when mixing types
        # Users can manually handle this if needed
        if skip_existing:
            print(
                "\n⚠️  Note: Skipping existing embeddings check when auto-detecting lyrics/metadata"
            )
            print(
                "    All songs will be processed. Duplicates will be handled via ON CONFLICT."
            )

        # Step 2: Generate embeddings in batches with parallel processing
        print(f"\nGenerating embeddings for {len(df):,} songs...")
        total_batches = (len(df) + batch_size - 1) // batch_size
        print(f"Processing in {total_batches:,} batches with 8 parallel workers...")

        embedding_client = EmbeddingClient()
        rate_limiter = RateLimiter(
            requests_per_minute=5800
        )  # Slightly under 6000 for safety

        # Thread-safe progress tracking
        progress_lock = Lock()
        processed_count = 0
        start_time = time.time()

        # Prepare all batches
        batch_tasks = []
        for batch_num in range(total_batches):
            start_idx = batch_num * batch_size
            end_idx = min((batch_num + 1) * batch_size, len(df))
            batch_df = df.slice(start_idx, end_idx - start_idx)
            batch_tasks.append((batch_num, batch_df))

        # Process batches in parallel
        all_results = [None] * total_batches
        total_errors = 0

        with ThreadPoolExecutor(max_workers=8) as executor:
            # Submit all tasks
            future_to_batch = {
                executor.submit(
                    process_batch_with_retry,
                    batch_num,
                    batch_df,
                    embedding_client,
                    rate_limiter,
                ): batch_num
                for batch_num, batch_df in batch_tasks
            }

            # Process completed tasks
            for future in as_completed(future_to_batch):
                if _interrupted:
                    print("\n⚠️  Cancelling remaining tasks...")
                    executor.shutdown(wait=False, cancel_futures=True)
                    break

                batch_num = future_to_batch[future]
                try:
                    result = future.result()
                    all_results[result["batch_num"]] = result

                    with progress_lock:
                        processed_count += 1
                        total_errors += result["errors"]

                        # Progress update every 10 batches
                        if (
                            processed_count % 10 == 0
                            or processed_count == total_batches
                        ):
                            elapsed_time = time.time() - start_time
                            successful = sum(
                                1 for r in all_results if r and r["success"]
                            )

                            # Calculate rates
                            batches_per_min = (
                                (processed_count / elapsed_time) * 60
                                if elapsed_time > 0
                                else 0
                            )
                            songs_processed = processed_count * batch_size
                            songs_per_min = (
                                (songs_processed / elapsed_time) * 60
                                if elapsed_time > 0
                                else 0
                            )

                            # Calculate ETA
                            remaining_batches = total_batches - processed_count
                            eta_seconds = (
                                (remaining_batches / batches_per_min) * 60
                                if batches_per_min > 0
                                else 0
                            )
                            eta_hours = eta_seconds / 3600
                            eta_mins = (eta_seconds % 3600) / 60

                            print(
                                f"  Progress: {processed_count}/{total_batches} batches "
                                f"({successful} successful, {total_errors} total errors) | "
                                f"Rate: {batches_per_min:.1f} batches/min ({songs_per_min:.0f} songs/min) | "
                                f"Elapsed: {elapsed_time / 60:.1f}m | "
                                f"ETA: {int(eta_hours)}h {int(eta_mins)}m"
                            )

                except Exception as e:
                    print(f"  Batch {batch_num + 1}: Unexpected error: {e}")
                    all_results[batch_num] = {
                        "batch_num": batch_num,
                        "embeddings": [],
                        "types": [],
                        "errors": batch_size,
                        "success": False,
                    }

        # Flatten results in order
        all_embeddings = []
        all_embedding_types = []
        for result in all_results:
            if result:
                all_embeddings.extend(result["embeddings"])
                all_embedding_types.extend(result["types"])

        successful_count = len([e for e in all_embeddings if e is not None])
        total_elapsed = time.time() - start_time

        print(f"\n✓ Generated {successful_count:,} embeddings")
        if total_errors > 0:
            print(f"⚠️  Failed to generate {total_errors:,} embeddings")
        print(
            f"⏱️  Total time: {total_elapsed / 60:.1f} minutes ({total_elapsed / 3600:.2f} hours)"
        )
        print(
            f"⚡ Average rate: {(processed_count / total_elapsed) * 60:.1f} batches/min, {(successful_count / total_elapsed) * 60:.0f} songs/min"
        )

        # Add embeddings and embedding types to DataFrame (only for the rows we processed)
        df_processed = df.head(len(all_embeddings))
        df_processed = df_processed.with_columns(
            [
                pl.Series("embedding", all_embeddings),
                pl.Series("embedding_type", all_embedding_types),
            ]
        )

        # Step 3: Store embeddings back to database with their respective types
        if not _interrupted or successful_count > 0:
            stored_count = store_embeddings_with_types(
                df_processed, postgres_url, batch_size=1000
            )
        else:
            stored_count = 0

        print("\n" + "=" * 80)
        if _interrupted:
            print("⚠️  PIPELINE INTERRUPTED")
        else:
            print("✓ PIPELINE COMPLETE")
        print("=" * 80)

        return {
            "total_songs": len(df),
            "embeddings_generated": successful_count,
            "embeddings_stored": stored_count,
            "errors": total_errors,
            "interrupted": _interrupted,
        }

    except KeyboardInterrupt:
        print("\n\n⚠️  Force quit detected. Progress may be lost.")
        return {
            "total_songs": len(df) if "df" in locals() else 0,
            "embeddings_generated": 0,
            "embeddings_stored": 0,
            "errors": 0,
            "interrupted": True,
        }
    finally:
        # Restore default signal handler
        signal.signal(signal.SIGINT, signal.SIG_DFL)
