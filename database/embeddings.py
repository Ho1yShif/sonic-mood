"""
Database operations for embeddings.
"""

from typing import Optional, Dict, Any, List
import signal
import time
import random
import os
from pathlib import Path
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


# ============================================================================
# Cache Management Functions
# ============================================================================


def cache_songs_to_parquet(
    parquet_path: str,
    table_name: str = None,
    postgres_url: str = None,
) -> int:
    """
    Cache all songs from database to a parquet file for faster subsequent reads.

    Args:
        parquet_path: Path to save the parquet file
        table_name: Database table to read from (default: songs)
        postgres_url: PostgreSQL connection URL

    Returns:
        Number of songs cached
    """
    print("\n" + "=" * 80)
    print("CACHING SONGS TO PARQUET FILE")
    print("=" * 80)

    parquet_file = Path(parquet_path).expanduser()
    parquet_file.parent.mkdir(parents=True, exist_ok=True)

    print("Reading all songs from database...")
    df = read_songs_from_postgres(
        table_name=table_name,
        postgres_url=postgres_url,
        start_from_song_id=None,  # Read all songs
    )

    print(f"\nWriting {len(df):,} songs to parquet file...")
    print(f"Location: {parquet_file}")

    df.write_parquet(parquet_file, compression="zstd")

    file_size_mb = parquet_file.stat().st_size / (1024 * 1024)
    print(f"✓ Cached {len(df):,} songs to parquet ({file_size_mb:.1f} MB)")
    print("=" * 80)

    return len(df)


def load_songs_from_parquet(
    parquet_path: str,
    start_from_song_id: Optional[int] = None,
    limit: Optional[int] = None,
) -> pl.DataFrame:
    """
    Load songs from parquet cache file.

    Args:
        parquet_path: Path to the parquet file
        start_from_song_id: Optional song_id to start from (for resume)
        limit: Optional limit on number of rows

    Returns:
        Polars DataFrame with songs data
    """
    parquet_file = Path(parquet_path).expanduser()

    if not parquet_file.exists():
        raise FileNotFoundError(f"Songs cache file not found: {parquet_file}")

    print(f"Loading songs from parquet cache: {parquet_file}")

    # Load parquet file
    df = pl.read_parquet(parquet_file)

    # Filter by start_from_song_id if provided
    if start_from_song_id is not None:
        print(f"Filtering to songs with song_id > {start_from_song_id}")
        df = df.filter(pl.col("song_id") > start_from_song_id)

    # Sort by song_id to ensure deterministic order
    df = df.sort("song_id")

    # Apply limit if provided
    if limit is not None:
        df = df.head(limit)

    print(f"✓ Loaded {len(df):,} songs from cache")

    # Count songs with/without lyrics
    lyrics_count = df.filter(
        (pl.col("lyrics").is_not_null()) & (pl.col("lyrics").str.strip_chars() != "")
    ).shape[0]
    metadata_count = df.shape[0] - lyrics_count
    print(f"  Songs with lyrics: {lyrics_count:,}")
    print(f"  Songs without lyrics (metadata only): {metadata_count:,}")

    return df


def get_max_cached_song_id(cache_dir: str) -> Optional[int]:
    """
    Find the maximum song_id that has been cached.

    Args:
        cache_dir: Directory containing cached embedding files

    Returns:
        Maximum song_id found, or None if directory is empty
    """
    cache_path = Path(cache_dir).expanduser()

    # Create directory if it doesn't exist
    cache_path.mkdir(parents=True, exist_ok=True)

    # List all .txt files and extract song_ids
    song_ids = []
    for file_path in cache_path.glob("*.txt"):
        try:
            song_id = int(file_path.stem)
            song_ids.append(song_id)
        except ValueError:
            # Skip files that don't have integer names
            continue

    if not song_ids:
        return None

    return max(song_ids)


def write_embedding_to_file(
    song_id: int, embedding: List[float], cache_dir: str
) -> bool:
    """
    Write an embedding to a cache file.

    Args:
        song_id: ID of the song
        embedding: Embedding vector as list of floats
        cache_dir: Directory to store cached embeddings

    Returns:
        True on success, False on error
    """
    try:
        cache_path = Path(cache_dir).expanduser()
        cache_path.mkdir(parents=True, exist_ok=True)

        file_path = cache_path / f"{song_id}.txt"

        # Write as comma-separated values
        csv_string = ",".join(map(str, embedding))
        file_path.write_text(csv_string)

        return True
    except Exception as e:
        print(f"  ⚠️  Error writing embedding for song_id {song_id}: {e}")
        return False


def read_embedding_from_file(song_id: int, cache_dir: str) -> Optional[List[float]]:
    """
    Read an embedding from a cache file.

    Args:
        song_id: ID of the song
        cache_dir: Directory containing cached embeddings

    Returns:
        Embedding vector as list of floats, or None if file doesn't exist or parsing fails
    """
    try:
        cache_path = Path(cache_dir).expanduser()
        file_path = cache_path / f"{song_id}.txt"

        if not file_path.exists():
            return None

        # Read and parse CSV
        csv_string = file_path.read_text()
        embedding = [float(x) for x in csv_string.split(",")]

        return embedding
    except Exception as e:
        print(f"  ⚠️  Error reading embedding for song_id {song_id}: {e}")
        return None


def load_embeddings_from_cache(
    cache_dir: str, song_ids: Optional[List[int]] = None
) -> pl.DataFrame:
    """
    Load embeddings from cache directory.

    Args:
        cache_dir: Directory containing cached embeddings
        song_ids: Optional list of specific song_ids to load (loads all if None)

    Returns:
        Polars DataFrame with columns: song_id, embedding
    """
    cache_path = Path(cache_dir).expanduser()

    if not cache_path.exists():
        print(f"Cache directory does not exist: {cache_path}")
        return pl.DataFrame({"song_id": [], "embedding": []})

    # Determine which files to load
    if song_ids is not None:
        files_to_load = [
            (cache_path / f"{song_id}.txt", song_id) for song_id in song_ids
        ]
    else:
        files_to_load = []
        for file_path in sorted(cache_path.glob("*.txt")):
            try:
                song_id = int(file_path.stem)
                files_to_load.append((file_path, song_id))
            except ValueError:
                continue

    # Load embeddings
    loaded_song_ids = []
    loaded_embeddings = []

    for i, (file_path, song_id) in enumerate(files_to_load):
        try:
            if file_path.exists():
                csv_string = file_path.read_text()
                embedding = [float(x) for x in csv_string.split(",")]
                loaded_song_ids.append(song_id)
                loaded_embeddings.append(embedding)

            # Show progress every 1000 files
            if (i + 1) % 1000 == 0:
                print(f"  Loaded {i + 1:,} embeddings from cache...")
        except Exception as e:
            print(f"  ⚠️  Error loading embedding for song_id {song_id}: {e}")
            continue

    if loaded_song_ids:
        print(f"✓ Loaded {len(loaded_song_ids):,} embeddings from cache")

    return pl.DataFrame({"song_id": loaded_song_ids, "embedding": loaded_embeddings})


def upload_cached_embeddings_to_database(
    cache_dir: str, postgres_url: str = None, batch_size: int = 500000
) -> int:
    """
    Upload cached embeddings to the database in batches.

    Args:
        cache_dir: Directory containing cached embeddings
        postgres_url: PostgreSQL connection URL (defaults to POSTGRES_URL env var)
        batch_size: Number of embeddings to upload per batch (default: 500K)

    Returns:
        Total number of embeddings uploaded
    """
    print("\n" + "=" * 80)
    print("UPLOADING CACHED EMBEDDINGS TO DATABASE")
    print("=" * 80)

    cache_path = Path(cache_dir).expanduser()

    # Get all cached song_ids
    all_song_ids = []
    for file_path in sorted(cache_path.glob("*.txt")):
        try:
            song_id = int(file_path.stem)
            all_song_ids.append(song_id)
        except ValueError:
            continue

    if not all_song_ids:
        print("No cached embeddings found.")
        return 0

    print(f"Found {len(all_song_ids):,} cached embeddings")
    print(f"Uploading in batches of {batch_size:,}...")

    total_uploaded = 0
    num_batches = (len(all_song_ids) + batch_size - 1) // batch_size

    for batch_num in range(num_batches):
        start_idx = batch_num * batch_size
        end_idx = min((batch_num + 1) * batch_size, len(all_song_ids))
        batch_song_ids = all_song_ids[start_idx:end_idx]

        print(
            f"\nBatch {batch_num + 1}/{num_batches} (song_ids {batch_song_ids[0]}-{batch_song_ids[-1]}):"
        )

        # Time loading from cache
        load_start = time.time()
        batch_df = load_embeddings_from_cache(cache_dir, batch_song_ids)
        load_time = time.time() - load_start
        print(f"  Load time: {load_time:.2f}s")

        if len(batch_df) == 0:
            print("  Skipping empty batch")
            continue

        # Time upload to database
        upload_start = time.time()
        uploaded = store_embeddings_copy_bulk(
            batch_df, postgres_url, show_progress=False
        )
        upload_time = time.time() - upload_start
        print(f"  Upload time: {upload_time:.2f}s")
        print(f"  Total time: {load_time + upload_time:.2f}s")

        total_uploaded += uploaded

    print("\n" + "=" * 80)
    print(f"✓ UPLOAD COMPLETE: {total_uploaded:,} embeddings")
    print("=" * 80)

    return total_uploaded


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

    for row in batch_df.iter_rows(named=True):
        song_name = row.get("song_name", "")
        band = row.get("band", "")
        lyrics = row.get("lyrics", "")

        # Always include song_name and band
        base_text = (
            f"{song_name} by {band}" if song_name and band else song_name or band
        )

        # If lyrics are available, append them
        if lyrics and str(lyrics).strip():
            text = f"{base_text}. {lyrics}"
        else:
            text = base_text

        texts_for_embedding.append(text if text else "")

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
                        "errors": len(texts_for_embedding),
                        "success": False,
                    }

    # Max retries exceeded
    return {
        "batch_num": batch_num,
        "embeddings": [None] * len(texts_for_embedding),
        "errors": len(texts_for_embedding),
        "success": False,
    }


def read_songs_from_postgres(
    table_name: str = None,
    limit: Optional[int] = None,
    postgres_url: str = None,
    start_from_song_id: Optional[int] = None,
) -> pl.DataFrame:
    """
    Read songs data from PostgreSQL database.

    Args:
        table_name: Name of the table to read from (defaults to songs table)
        limit: Optional limit on number of rows to read
        postgres_url: PostgreSQL connection URL (defaults to POSTGRES_URL env var)
        start_from_song_id: Optional song_id to start from (exclusive, for resume functionality)

    Returns:
        Polars DataFrame containing songs data with song_id, song_name, band, and lyrics columns
    """
    # Always default to table with lyrics
    if table_name is None:
        table_name = "songs"

    # Always include lyrics column
    selected_columns = ["song_id", "song_name", "band", "lyrics"]
    columns_str = ", ".join(selected_columns)

    print(f"Reading songs from table '{table_name}'...")
    if start_from_song_id is not None:
        print(f"Starting from song_id > {start_from_song_id} (resume mode)")
    print("Embedding strategy: song_name + band + lyrics (if available)")
    print(f"Selecting columns: {columns_str}")

    query = f"SELECT {columns_str} FROM {table_name}"
    if start_from_song_id is not None:
        query += f" WHERE song_id > {start_from_song_id}"
    query += " ORDER BY song_id ASC"
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
    postgres_url: str = None,
) -> pl.DataFrame:
    """
    Check which songs already have embeddings and filter them out.

    Args:
        df: DataFrame containing songs with 'song_id' column
        postgres_url: PostgreSQL connection URL (defaults to POSTGRES_URL env var)

    Returns:
        DataFrame filtered to only songs that don't have embeddings yet
    """
    print("\nChecking for existing embeddings...")

    with get_connection(postgres_url) as conn:
        result = conn.execute(
            text("""
                SELECT DISTINCT song_id 
                FROM song_embeddings
            """)
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
    postgres_url: str = None,
    batch_size: int = 100,
) -> int:
    """
    DEPRECATED: Use store_embeddings_batch instead.

    Store embeddings in the song_embeddings table.
    Each insert has its own transaction to prevent cascade failures.

    Args:
        df: DataFrame containing songs with embeddings (must have 'song_id' and 'embedding' columns)
        postgres_url: PostgreSQL connection URL (defaults to POSTGRES_URL env var)
        batch_size: Number of embeddings to insert per batch

    Returns:
        Number of embeddings stored
    """
    print(
        "\n⚠️  WARNING: store_embeddings is deprecated. Use store_embeddings_batch instead."
    )
    print("Storing embeddings in song_embeddings table...")

    # Filter out rows without embeddings
    df_with_embeddings = df.filter(pl.col("embedding").is_not_null())

    print(f"Inserting {len(df_with_embeddings):,} embeddings...")

    stored_count = 0
    error_count = 0
    with get_connection(postgres_url) as conn:
        for i, row in enumerate(df_with_embeddings.iter_rows(named=True)):
            if _interrupted:
                print(
                    f"\n⚠️  Interrupted. Stored {stored_count:,} embeddings before interruption."
                )
                break

            try:
                song_id = row["song_id"]
                embedding = row["embedding"]

                # Convert embedding list to pgvector format
                embedding_str = "[" + ",".join(map(str, embedding)) + "]"

                # Each insert gets its own transaction
                with conn.begin():
                    conn.execute(
                        text("""
                            INSERT INTO song_embeddings 
                            (song_id, embedding)
                            VALUES (:song_id, CAST(:embedding AS vector))
                            ON CONFLICT (song_id) 
                            DO UPDATE SET 
                                embedding = EXCLUDED.embedding,
                                created_at = NOW()
                        """),
                        {
                            "song_id": song_id,
                            "embedding": embedding_str,
                        },
                    )
                stored_count += 1
            except Exception as e:
                error_count += 1
                if error_count <= 10:
                    print(f"    ⚠️  Error storing embedding for song_id {song_id}: {e}")
                elif error_count == 11:
                    print("    ⚠️  Suppressing further error messages...")

            if (i + 1) % batch_size == 0:
                print(
                    f"  Progress: {i + 1:,}/{len(df_with_embeddings):,} embeddings stored ({error_count} errors)..."
                )

    if not _interrupted:
        # Verify the data was stored
        with get_connection(postgres_url) as conn:
            result = conn.execute(
                text("""
                    SELECT embedding_type, COUNT(*) 
                    FROM song_embeddings
                    GROUP BY embedding_type
                """)
            )
            counts = result.fetchall()
            print(
                f"✓ Successfully stored {stored_count:,} embeddings ({error_count} errors)"
            )
            print("  Total counts by type:")
            for embedding_type, count in counts:
                print(f"    {embedding_type}: {count:,}")

    return stored_count


def store_embeddings_copy_bulk(
    df: pl.DataFrame,
    postgres_url: str = None,
    show_progress: bool = True,
    progress_interval: int = 10000,
) -> int:
    """
    Store embeddings using PostgreSQL COPY command with binary format for maximum speed.
    This is 10-100x faster than individual INSERT statements.

    Uses psycopg3 with binary COPY format for optimal performance.
    Creates a temporary table, copies data in bulk, then merges into final table.

    Args:
        df: DataFrame containing songs with embeddings (must have 'song_id' and 'embedding' columns)
        postgres_url: PostgreSQL connection URL (defaults to POSTGRES_URL env var)
        show_progress: If True, show progress updates during copy
        progress_interval: Show progress every N rows

    Returns:
        Number of embeddings stored
    """
    try:
        import psycopg
        from pgvector.psycopg import register_vector
    except ImportError:
        print("⚠️  psycopg3 not found. Falling back to slower INSERT method.")
        print("   Install with: pip install 'psycopg[binary]>=3.0'")
        return store_embeddings_batch(df, postgres_url)

    print("\nStoring embeddings using COPY bulk insert (fast method)...")

    # Filter out rows without embeddings
    df_with_embeddings = df.filter(pl.col("embedding").is_not_null())
    total_rows = len(df_with_embeddings)

    if total_rows == 0:
        print("No embeddings to store.")
        return 0

    print(f"Preparing {total_rows:,} embeddings for bulk insert...")

    # Get connection URL
    url = postgres_url or os.environ.get("POSTGRES_URL")
    if not url:
        raise ValueError("No PostgreSQL URL provided and POSTGRES_URL env var not set")

    # Convert embeddings to proper format
    song_ids = df_with_embeddings["song_id"].to_list()
    embeddings_list = df_with_embeddings["embedding"].to_list()

    print(
        f"  Embedding dimensions: {len(embeddings_list)} embeddings x {len(embeddings_list[0])} dimensions"
    )

    try:
        # Connect with psycopg3
        with psycopg.connect(url, autocommit=False) as conn:
            # Register vector type
            register_vector(conn)

            # Create temporary table for bulk insert
            with conn.cursor() as cur:
                print("  Creating temporary table...")
                cur.execute("""
                    CREATE TEMP TABLE temp_embeddings (
                        song_id INTEGER NOT NULL,
                        embedding vector(2000) NOT NULL
                    ) ON COMMIT DROP
                """)

                # Use COPY with text format and proper vector formatting
                print(f"  Bulk copying {total_rows:,} embeddings...")
                with cur.copy(
                    "COPY temp_embeddings (song_id, embedding) FROM STDIN"
                ) as copy:
                    for i, (song_id, embedding) in enumerate(
                        zip(song_ids, embeddings_list)
                    ):
                        # Format embedding with brackets for pgvector
                        embedding_str = "[" + ",".join(map(str, embedding)) + "]"
                        # Write as tab-separated values
                        copy.write_row([int(song_id), embedding_str])

                        # Show progress
                        if show_progress and (i + 1) % progress_interval == 0:
                            print(
                                f"    Progress: {i + 1:,}/{total_rows:,} rows copied...",
                                flush=True,
                            )

                if show_progress:
                    print(f"    ✓ Copied all {total_rows:,} rows")

                # Merge into final table using INSERT ... ON CONFLICT
                print("  Merging into song_embeddings table...")
                cur.execute("""
                    INSERT INTO song_embeddings (song_id, embedding, created_at)
                    SELECT song_id, embedding, NOW()
                    FROM temp_embeddings
                    ON CONFLICT (song_id) 
                    DO UPDATE SET 
                        embedding = EXCLUDED.embedding,
                        created_at = NOW()
                """)

                merged_count = cur.rowcount

                # Commit the transaction
                conn.commit()

                print(
                    f"✓ Successfully stored {merged_count:,} embeddings using COPY bulk insert"
                )

                # Verify
                cur.execute("SELECT COUNT(*) FROM song_embeddings")
                total_in_db = cur.fetchone()[0]
                print(f"  Total embeddings in database: {total_in_db:,}")

                return merged_count

    except Exception as e:
        print(f"❌ Error during COPY bulk insert: {e}")
        print("   Falling back to slower INSERT method...")
        return store_embeddings_batch(df, postgres_url)


def store_embeddings_batch(
    df: pl.DataFrame,
    postgres_url: str = None,
    batch_size: int = 1000,
) -> int:
    """
    Store embeddings in the song_embeddings table.
    Each insert has its own transaction to prevent cascade failures.

    Args:
        df: DataFrame containing songs with embeddings (must have 'song_id' and 'embedding' columns)
        postgres_url: PostgreSQL connection URL (defaults to POSTGRES_URL env var)
        batch_size: Number of embeddings to insert per batch (default 1000)

    Returns:
        Number of embeddings stored
    """
    print("\nStoring embeddings in song_embeddings table...")

    # Filter out rows without embeddings
    df_with_embeddings = df.filter(pl.col("embedding").is_not_null())

    print(
        f"Inserting {len(df_with_embeddings):,} embeddings in batches of {batch_size}..."
    )

    stored_count = 0
    error_count = 0
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

            # Use individual transactions per insert to prevent cascade failures
            for row in batch_df.iter_rows(named=True):
                try:
                    song_id = row["song_id"]
                    embedding = row["embedding"]
                    embedding_str = "[" + ",".join(map(str, embedding)) + "]"

                    # Each insert gets its own transaction
                    with conn.begin():
                        conn.execute(
                            text("""
                                INSERT INTO song_embeddings 
                                (song_id, embedding)
                                VALUES (:song_id, CAST(:embedding AS vector))
                                ON CONFLICT (song_id) 
                                DO UPDATE SET 
                                    embedding = EXCLUDED.embedding,
                                    created_at = NOW()
                            """),
                            {
                                "song_id": song_id,
                                "embedding": embedding_str,
                            },
                        )
                    stored_count += 1
                except Exception as e:
                    error_count += 1
                    # Only print first 10 errors to avoid spam
                    if error_count <= 10:
                        print(
                            f"    ⚠️  Error storing embedding for song_id {song_id}: {e}"
                        )
                    elif error_count == 11:
                        print("    ⚠️  Suppressing further error messages...")

            if (batch_num + 1) % 10 == 0 or batch_num + 1 == total_batches:
                print(
                    f"  Progress: {stored_count:,}/{len(df_with_embeddings):,} embeddings stored ({error_count} errors)..."
                )

    if not _interrupted:
        # Verify the data was stored
        with get_connection(postgres_url) as conn:
            result = conn.execute(
                text("""
                    SELECT COUNT(*) 
                    FROM song_embeddings
                """)
            )
            count = result.fetchone()[0]
            print(
                f"✓ Successfully stored {stored_count:,} embeddings ({error_count} errors)"
            )
            print(f"  Total embeddings in database: {count:,}")

    return stored_count


def generate_and_store_all_embeddings(
    table_name: str = None,
    limit: Optional[int] = None,
    postgres_url: str = None,
    skip_existing: bool = True,
    batch_size: int = 2048,
    checkpoint_interval: int = 100,
) -> Dict[str, Any]:
    """
    Complete pipeline: read songs, generate embeddings, store in database.

    Uses batch processing to send up to 2048 songs per API call for efficiency.
    According to Fireworks AI docs: https://docs.fireworks.ai/api-reference/creates-an-embedding-vector-representing-the-input-text
    - Can send up to 2048 strings per request
    - Rate limit: 6000 requests per minute

    For each song, embeds song_name and band. If lyrics are available, they are also included in the embedding.
    All embeddings are stored in the same field - no separate embedding types.

    Supports automatic resume: If the process is interrupted, already-processed embeddings are stored,
    and re-running will skip them automatically (when skip_existing=True).

    Args:
        table_name: Name of the table to process (defaults to songs)
        limit: Optional limit on number of songs to process
        postgres_url: PostgreSQL connection URL (defaults to POSTGRES_URL env var)
        skip_existing: If True, skip songs that already have embeddings (enables auto-resume)
        batch_size: Number of songs to process per API call (max 2048, default 2048)
        checkpoint_interval: Store embeddings to DB every N batches (default 100, reduces memory)

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
    print("Embedding strategy: song_name + band + lyrics (if available)")
    print(f"Model: {'qwen3-embedding-8b'}")
    print(f"Batch size: {batch_size} songs per API call")

    try:
        # Step 1: Read songs from database
        df = read_songs_from_postgres(table_name, limit, postgres_url)

        # Count how many have lyrics vs metadata only
        lyrics_count = df.filter(
            (pl.col("lyrics").is_not_null())
            & (pl.col("lyrics").str.strip_chars() != "")
        ).shape[0]
        metadata_count = df.shape[0] - lyrics_count
        print(f"\nSongs with lyrics: {lyrics_count:,}")
        print(f"Songs without lyrics (will use metadata only): {metadata_count:,}")

        # Check for existing embeddings to enable auto-resume
        if skip_existing:
            print("\n🔍 Checking for existing embeddings to enable auto-resume...")
            with get_connection(postgres_url) as conn:
                # Get all song_ids that already have embeddings (any type)
                result = conn.execute(
                    text("""
                        SELECT DISTINCT song_id 
                        FROM song_embeddings
                    """)
                )
                existing_song_ids = {row[0] for row in result.fetchall()}

            if existing_song_ids:
                original_count = len(df)
                df = df.filter(~pl.col("song_id").is_in(list(existing_song_ids)))
                skipped = original_count - len(df)
                print(
                    f"  ✓ Found {len(existing_song_ids):,} songs with existing embeddings"
                )
                print(
                    f"  ✓ Skipping {skipped:,} songs, will process {len(df):,} new songs"
                )

                if len(df) == 0:
                    print("\n✓ All songs already have embeddings! Nothing to process.")
                    return {
                        "total_songs": original_count,
                        "embeddings_generated": 0,
                        "embeddings_stored": 0,
                        "errors": 0,
                        "interrupted": False,
                    }
            else:
                print(
                    f"  ✓ No existing embeddings found, will process all {len(df):,} songs"
                )

        # Step 2: Generate embeddings in batches with parallel processing and checkpointing
        print(f"\nGenerating embeddings for {len(df):,} songs...")
        total_batches = (len(df) + batch_size - 1) // batch_size
        print(f"Processing in {total_batches:,} batches with 8 parallel workers...")
        print(
            f"Checkpointing every {checkpoint_interval} batches to reduce memory usage"
        )

        embedding_client = EmbeddingClient()
        rate_limiter = RateLimiter(
            requests_per_minute=5800
        )  # Slightly under 6000 for safety

        # Thread-safe progress tracking
        progress_lock = Lock()
        processed_count = 0
        start_time = time.time()
        total_stored = 0
        total_errors = 0

        # Prepare all batches
        batch_tasks = []
        for batch_num in range(total_batches):
            start_idx = batch_num * batch_size
            end_idx = min((batch_num + 1) * batch_size, len(df))
            batch_df = df.slice(start_idx, end_idx - start_idx)
            batch_tasks.append((batch_num, batch_df))

        # Process batches in parallel with periodic checkpointing
        checkpoint_results = []
        all_results = [None] * total_batches

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
                    checkpoint_results.append(result)

                    with progress_lock:
                        processed_count += 1
                        total_errors += result["errors"]

                        # Checkpoint: Store embeddings periodically to reduce memory
                        if (
                            processed_count % checkpoint_interval == 0
                            or processed_count == total_batches
                        ):
                            if checkpoint_results:
                                try:
                                    # Flatten checkpoint results
                                    cp_embeddings = []
                                    cp_song_ids = []

                                    for cp_result in checkpoint_results:
                                        if cp_result and cp_result["success"]:
                                            # Get the original batch to retrieve song_ids
                                            cp_batch_num = cp_result["batch_num"]
                                            start_idx = cp_batch_num * batch_size
                                            end_idx = min(
                                                (cp_batch_num + 1) * batch_size, len(df)
                                            )
                                            cp_batch_df = df.slice(
                                                start_idx, end_idx - start_idx
                                            )

                                            cp_song_ids.extend(
                                                cp_batch_df["song_id"].to_list()
                                            )
                                            cp_embeddings.extend(
                                                cp_result["embeddings"]
                                            )

                                    # Create mini DataFrame and store using fast COPY bulk insert
                                    if cp_embeddings:
                                        cp_df = pl.DataFrame(
                                            {
                                                "song_id": cp_song_ids,
                                                "embedding": cp_embeddings,
                                            }
                                        )
                                        stored = store_embeddings_copy_bulk(
                                            cp_df, postgres_url, show_progress=False
                                        )
                                        total_stored += stored

                                        # Clear checkpoint results to free memory
                                        checkpoint_results = []
                                        print(
                                            f"  💾 Checkpoint: Stored {total_stored:,} embeddings to database"
                                        )
                                except Exception as e:
                                    print(f"  ⚠️  Checkpoint storage error: {e}")
                                    print(
                                        "  ⚠️  Will retry at next checkpoint. Progress not lost."
                                    )
                                    # Don't clear checkpoint_results, will retry next time

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
                        "errors": batch_size,
                        "success": False,
                    }
                    checkpoint_results.append(all_results[batch_num])

        # Calculate final statistics
        successful_count = sum(
            1
            for r in all_results
            if r and r["success"]
            for e in r["embeddings"]
            if e is not None
        )
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
        print(f"💾 Total stored: {total_stored:,} embeddings in database")

        # Note: Embeddings are already stored via checkpointing
        stored_count = total_stored

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
