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
_interrupt_count = 0


def signal_handler(signum, frame):
    """Handle keyboard interrupt (Ctrl+C)."""
    global _interrupted, _interrupt_count
    _interrupt_count += 1

    if _interrupt_count == 1:
        _interrupted = True
        print(
            "\n\n⚠️  Keyboard interrupt received. Finishing current operations and exiting..."
        )
        print("⚠️  Press Ctrl+C again to force quit immediately (may lose progress)")
    else:
        print("\n\n⚠️  Force quit! Exiting immediately...")
        import sys

        sys.exit(1)


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


def _scan_cache_directory(cache_path: Path) -> List[Path]:
    """
    Scan cache directory and collect all embedding files.
    Shows progress every 100K files to avoid appearing hung.
    """
    print(f"\n📂 Scanning cache directory: {cache_path}")
    print("  Collecting embedding files (progress shown every 100K)...")

    all_files = []
    scan_start = time.time()

    for i, file_path in enumerate(cache_path.glob("*.txt")):
        try:
            int(file_path.stem)  # Validate it's a song_id
            all_files.append(file_path)

            if (i + 1) % 100000 == 0:
                elapsed = time.time() - scan_start
                rate = (i + 1) / elapsed
                print(f"  → Scanned {i + 1:,} files ({rate:.0f} files/sec)...")
        except ValueError:
            continue

    scan_time = time.time() - scan_start
    total_scanned = i + 1 if "i" in locals() else 0

    print(f"\n✓ Scan complete in {scan_time:.1f}s")
    print(f"  Total files found: {len(all_files):,}")

    return all_files


def _split_files_among_workers(files: List[Path], num_workers: int) -> List[List[Path]]:
    """Split files evenly among workers."""
    files_per_worker = len(files) // num_workers
    worker_chunks = []

    for i in range(num_workers):
        start = i * files_per_worker
        end = start + files_per_worker if i < num_workers - 1 else len(files)
        worker_chunks.append(files[start:end])

    return worker_chunks


def _calculate_progress_stats(
    chunk_idx: int,
    total_chunks: int,
    files_processed: int,
    total_files: int,
    total_uploaded: int,
    overall_start: float,
) -> dict:
    """Calculate progress statistics including ETA."""
    elapsed = time.time() - overall_start
    overall_rate = total_uploaded / elapsed if elapsed > 0 else 0
    percent_complete = ((chunk_idx + 1) / total_chunks) * 100

    remaining_files = total_files - files_processed
    eta_seconds = remaining_files / overall_rate if overall_rate > 0 else 0
    eta_hours = eta_seconds / 3600

    return {
        "elapsed_min": elapsed / 60,
        "overall_rate": overall_rate,
        "percent_complete": percent_complete,
        "eta_hours": eta_hours,
    }


def upload_cached_embeddings_to_database(
    cache_dir: str,
    postgres_url: str = None,
    chunk_size: int = 5000,
    num_workers: int = 5,
) -> int:
    """
    Upload cached embeddings using parallel workers with chunked processing.

    Strategy:
    - Scans all files once (with progress every 100K files)
    - Processes in chunks of 5K to prevent connection timeouts
    - Each chunk uses 5 parallel workers with separate DB connections
    - Each worker uses COPY command for fast bulk upload

    Args:
        cache_dir: Directory containing cached embeddings
        postgres_url: PostgreSQL connection URL (defaults to POSTGRES_URL env var)
        chunk_size: Number of files per chunk (default: 5K - keeps DB connections alive)
        num_workers: Number of parallel workers per chunk (default: 5)

    Returns:
        Total number of embeddings uploaded
    """
    # Reset interrupt handling
    global _interrupted, _interrupt_count
    _interrupted = False
    _interrupt_count = 0
    signal.signal(signal.SIGINT, signal_handler)

    # Print configuration
    print("\n" + "=" * 80)
    print("UPLOADING CACHED EMBEDDINGS (Parallel Chunked)")
    print("=" * 80)
    print(f"Configuration:")
    print(f"  Chunk size: {chunk_size:,} files per chunk")
    print(f"  Workers: {num_workers} parallel workers per chunk")
    print("=" * 80)

    # Scan directory for all embedding files
    cache_path = Path(cache_dir).expanduser()
    all_files = _scan_cache_directory(cache_path)

    if not all_files:
        print("\n✓ No embedding files found!")
        return 0

    # Process files in chunks
    total_chunks = (len(all_files) + chunk_size - 1) // chunk_size
    print(f"\n🚀 Starting upload: {total_chunks} chunks")
    print("=" * 80)

    total_uploaded = 0
    overall_start = time.time()

    for chunk_idx in range(total_chunks):
        # Extract chunk
        start_idx = chunk_idx * chunk_size
        end_idx = min(start_idx + chunk_size, len(all_files))
        chunk_files = all_files[start_idx:end_idx]

        # Print chunk header
        print(f"\n📦 CHUNK {chunk_idx + 1}/{total_chunks}")
        print(f"  Files: {len(chunk_files):,}")
        print(f"  Range: {chunk_files[0].stem} to {chunk_files[-1].stem}")

        # Split among workers
        worker_chunks = _split_files_among_workers(chunk_files, num_workers)
        for i, wc in enumerate(worker_chunks):
            print(f"    Worker {i + 1}: {len(wc):,} files")

        # Upload chunk with parallel workers
        print(f"\n  ⚙️  Processing...")
        chunk_start = time.time()
        chunk_uploaded = _upload_chunk_parallel(
            worker_chunks, postgres_url, chunk_idx + 1, num_workers
        )
        chunk_time = time.time() - chunk_start
        total_uploaded += chunk_uploaded

        # Print chunk summary
        chunk_rate = chunk_uploaded / chunk_time if chunk_time > 0 else 0
        print(f"\n  ✓ Chunk {chunk_idx + 1} complete:")
        print(
            f"    Uploaded: {chunk_uploaded:,} in {chunk_time:.1f}s ({chunk_rate:.1f} rows/s)"
        )

        # Print overall progress
        stats = _calculate_progress_stats(
            chunk_idx,
            total_chunks,
            end_idx,
            len(all_files),
            total_uploaded,
            overall_start,
        )
        print(
            f"\n  📊 Overall: {end_idx:,}/{len(all_files):,} files ({stats['percent_complete']:.1f}%)"
        )
        print(f"    Total uploaded: {total_uploaded:,}")
        print(f"    Elapsed: {stats['elapsed_min']:.1f} min")
        print(f"    Rate: {stats['overall_rate']:.1f} rows/s")
        print(f"    ETA: {stats['eta_hours']:.1f} hours")
        print("=" * 80)

        # Check for interrupt
        if _interrupted:
            print("\n⚠️  Interrupt detected. Exiting gracefully...")
            break

    # Final summary
    elapsed_total = time.time() - overall_start
    avg_rate = total_uploaded / elapsed_total if elapsed_total > 0 else 0

    print("\n" + "=" * 80)
    if _interrupted:
        print(f"⚠️  UPLOAD INTERRUPTED")
        print(f"  Uploaded before interruption: {total_uploaded:,} embeddings")
    else:
        print(f"✓ UPLOAD COMPLETE")
        print(f"  Uploaded: {total_uploaded:,} embeddings")
    print(
        f"  Duration: {elapsed_total / 60:.1f} min ({elapsed_total / 3600:.1f} hours)"
    )
    print(f"  Average rate: {avg_rate:.1f} rows/sec")
    print("=" * 80)

    # Restore default signal handler
    signal.signal(signal.SIGINT, signal.SIG_DFL)

    return total_uploaded


def _read_and_format_embedding(file_path: Path) -> tuple:
    """
    Read embedding file and format for pgvector.

    Returns:
        Tuple of (song_id, embedding_literal) or (None, None) on error
    """
    try:
        song_id = int(file_path.stem)

        with open(file_path, "r") as f:
            embedding_str = f.read().strip()
            embedding = [float(x) for x in embedding_str.split(",")]

        # Format as pgvector literal with brackets
        embedding_literal = "[" + ",".join(map(str, embedding)) + "]"
        return song_id, embedding_literal

    except Exception:
        return None, None


def _upload_files_with_copy(
    copy_context, file_paths: List[Path], worker_id: int, start_time: float
) -> tuple:
    """
    Upload files using COPY context.

    CRITICAL: NO PRINT STATEMENTS during COPY - they cause "reentrant call" errors!

    Returns:
        Tuple of (rows_uploaded, errors)
    """
    rows_uploaded = 0
    errors = 0
    error_files = []  # Collect errors to report after COPY finishes

    for i, file_path in enumerate(file_paths, 1):
        song_id, embedding_literal = _read_and_format_embedding(file_path)

        if song_id is None:
            errors += 1
            if errors <= 5:  # Only collect first 5 errors
                error_files.append(str(file_path))
            continue

        copy_context.write_row((song_id, embedding_literal))
        rows_uploaded += 1

    return rows_uploaded, errors


def _upload_worker(worker_id: int, file_paths: List[Path], postgres_url: str) -> dict:
    """
    Worker function for parallel upload - each worker has its own DB connection.

    Args:
        worker_id: Worker identifier
        file_paths: List of file paths for this worker
        postgres_url: Database connection URL

    Returns:
        Dictionary with results: {rows_uploaded, errors, duration}
    """
    import psycopg

    result = {"worker_id": worker_id, "rows_uploaded": 0, "errors": 0, "duration": 0}
    start_time = time.time()

    try:
        print(f"      Worker {worker_id}: Connecting to database...")
        conn_start = time.time()
        # Add connection settings to prevent timeouts during long operations
        conn_params = {
            "connect_timeout": 30,
            "keepalives": 1,
            "keepalives_idle": 30,
            "keepalives_interval": 10,
            "keepalives_count": 5,
        }
        with psycopg.connect(postgres_url, **conn_params) as conn:
            conn_time = time.time() - conn_start
            print(f"      Worker {worker_id}: Connected in {conn_time:.2f}s")

            print(
                f"      Worker {worker_id}: Starting COPY for {len(file_paths):,} files..."
            )
            copy_start = time.time()

            with conn.cursor() as cur:
                # CRITICAL: NO PRINT STATEMENTS inside COPY context - causes errors!
                with cur.copy(
                    "COPY song_embeddings (song_id, embedding) FROM STDIN"
                ) as copy:
                    rows_uploaded, errors = _upload_files_with_copy(
                        copy, file_paths, worker_id, start_time
                    )
                    result["rows_uploaded"] = rows_uploaded
                    result["errors"] = errors

            copy_time = time.time() - copy_start
            print(
                f"      Worker {worker_id}: COPY complete in {copy_time:.2f}s, committing..."
            )

            commit_start = time.time()
            conn.commit()
            commit_time = time.time() - commit_start
            print(f"      Worker {worker_id}: Committed in {commit_time:.2f}s")

    except Exception as e:
        error_msg = str(e)
        print(f"    ❌ Worker {worker_id} failed: {error_msg[:200]}")

        # Log connection-specific errors
        if "timeout" in error_msg.lower():
            print(f"       → Connection/query timeout - consider reducing chunk size")
        elif "connection" in error_msg.lower() or "socket" in error_msg.lower():
            print(f"       → Connection lost - may need keepalive or smaller chunks")

        # Don't print full traceback for known errors to reduce noise
        if not any(x in error_msg.lower() for x in ["connection", "timeout", "socket"]):
            import traceback

            traceback.print_exc()

    result["duration"] = time.time() - start_time
    return result


def _upload_chunk_parallel(
    worker_chunks: List[List[Path]], postgres_url: str, chunk_num: int, num_workers: int
) -> int:
    """
    Upload a chunk using parallel workers.

    Args:
        worker_chunks: List of file lists, one per worker
        postgres_url: Database connection URL
        chunk_num: Chunk number (for logging)
        num_workers: Number of workers

    Returns:
        Total rows uploaded by all workers
    """
    global _interrupted

    url = postgres_url or os.environ.get("POSTGRES_URL")

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Submit all workers
        futures = []
        for i, worker_files in enumerate(worker_chunks):
            future = executor.submit(_upload_worker, i + 1, worker_files, url)
            futures.append(future)

        # Collect results
        total_uploaded = 0
        total_errors = 0

        try:
            for future in as_completed(
                futures, timeout=300
            ):  # 5 min timeout per worker
                if _interrupted:
                    print(
                        "\n    ⚠️  Interrupt detected, cancelling remaining workers..."
                    )
                    for f in futures:
                        f.cancel()
                    break

                result = future.result()
                total_uploaded += result["rows_uploaded"]
                total_errors += result["errors"]
                rate = (
                    result["rows_uploaded"] / result["duration"]
                    if result["duration"] > 0
                    else 0
                )
                print(
                    f"    ✓ Worker {result['worker_id']} complete: "
                    f"{result['rows_uploaded']:,} rows in {result['duration']:.1f}s "
                    f"({rate:.1f} rows/s)"
                )
        except TimeoutError:
            print(
                "\n    ⚠️  Workers timed out after 5 minutes - may need smaller chunks"
            )
            for f in futures:
                f.cancel()

        if total_errors > 0:
            print(f"    ⚠️  Total errors across all workers: {total_errors}")

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


def _extract_embeddings_from_checkpoint(
    checkpoint_results: list, df: pl.DataFrame, batch_size: int
) -> tuple:
    """
    Extract song_ids and embeddings from checkpoint results.

    Returns:
        Tuple of (song_ids, embeddings)
    """
    song_ids = []
    embeddings = []

    for result in checkpoint_results:
        if not result or not result["success"]:
            continue

        # Get the original batch to retrieve song_ids
        batch_num = result["batch_num"]
        start_idx = batch_num * batch_size
        end_idx = min((batch_num + 1) * batch_size, len(df))
        batch_df = df.slice(start_idx, end_idx - start_idx)

        song_ids.extend(batch_df["song_id"].to_list())
        embeddings.extend(result["embeddings"])

    return song_ids, embeddings


def _store_checkpoint(song_ids: list, embeddings: list, postgres_url: str) -> int:
    """
    Store checkpoint embeddings to database.

    Returns:
        Number of rows stored, or 0 on error
    """
    if not embeddings:
        return 0

    try:
        df = pl.DataFrame({"song_id": song_ids, "embedding": embeddings})
        return store_embeddings_copy_bulk(df, postgres_url, show_progress=False)
    except Exception as e:
        print(f"  ⚠️  Checkpoint storage error: {e}")
        print("  ⚠️  Will retry at next checkpoint. Progress not lost.")
        return 0


def _calculate_and_print_progress(
    processed_count: int,
    total_batches: int,
    all_results: list,
    total_errors: int,
    batch_size: int,
    start_time: float,
):
    """Calculate and print progress statistics."""
    elapsed_time = time.time() - start_time
    successful = sum(1 for r in all_results if r and r["success"])

    # Calculate rates
    batches_per_min = (processed_count / elapsed_time) * 60 if elapsed_time > 0 else 0
    songs_processed = processed_count * batch_size
    songs_per_min = (songs_processed / elapsed_time) * 60 if elapsed_time > 0 else 0

    # Calculate ETA
    remaining_batches = total_batches - processed_count
    eta_seconds = (
        (remaining_batches / batches_per_min) * 60 if batches_per_min > 0 else 0
    )
    eta_hours = int(eta_seconds / 3600)
    eta_mins = int((eta_seconds % 3600) / 60)

    print(
        f"  Progress: {processed_count}/{total_batches} batches "
        f"({successful} successful, {total_errors} total errors) | "
        f"Rate: {batches_per_min:.1f} batches/min ({songs_per_min:.0f} songs/min) | "
        f"Elapsed: {elapsed_time / 60:.1f}m | "
        f"ETA: {eta_hours}h {eta_mins}m"
    )


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
    if table_name is None:
        table_name = "songs"

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
                    INSERT INTO song_embeddings (song_id, embedding)
                    SELECT song_id, embedding
                    FROM temp_embeddings
                    ON CONFLICT (song_id) 
                    DO UPDATE SET 
                        embedding = EXCLUDED.embedding
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
    batch_size: int = 100000,
) -> int:
    """
    Store embeddings in the song_embeddings table using COPY bulk inserts.
    Uses a fallback strategy: tries 100K batches with COPY, then 10K, then 1K if failures occur.
    Falls back to individual INSERTs only as last resort.

    Args:
        df: DataFrame containing songs with embeddings (must have 'song_id' and 'embedding' columns)
        postgres_url: PostgreSQL connection URL (defaults to POSTGRES_URL env var)
        batch_size: Initial batch size (default 100000)

    Returns:
        Number of embeddings stored
    """
    # Try psycopg3 first for COPY method
    try:
        import psycopg
        from pgvector.psycopg import register_vector

        use_copy = True
    except ImportError:
        print("⚠️  psycopg3 not found. Using fallback INSERT method (slower).")
        print("   Install with: pip install 'psycopg[binary]>=3.0' pgvector")
        use_copy = False

    print("\nStoring embeddings in song_embeddings table...")

    # Filter out rows without embeddings
    df_with_embeddings = df.filter(pl.col("embedding").is_not_null())

    if use_copy:
        print(
            f"Inserting {len(df_with_embeddings):,} embeddings using COPY with fallback strategy..."
        )
    else:
        print(
            f"Inserting {len(df_with_embeddings):,} embeddings using INSERT with fallback strategy..."
        )

    stored_count = 0
    error_count = 0

    # Get connection URL
    url = postgres_url or os.environ.get("POSTGRES_URL")
    if not url:
        raise ValueError("No PostgreSQL URL provided and POSTGRES_URL env var not set")

    def _copy_batch(
        psycopg_conn, batch_data: list, batch_desc: str = ""
    ) -> tuple[int, int]:
        """
        Use COPY to insert a batch of data. Returns (success_count, error_count).
        Requires psycopg3 connection.
        """
        if not batch_data:
            return 0, 0

        batch_start_time = time.time()
        cursor = None
        try:
            print(
                f"    → Starting COPY operation for {len(batch_data):,} records{batch_desc}"
            )

            # Create a cursor
            cursor = psycopg_conn.cursor()

            # Step 1: Drop any existing temp table (cleanup from previous errors)
            try:
                cursor.execute("DROP TABLE IF EXISTS temp_batch_embeddings")
                psycopg_conn.commit()
            except Exception:
                try:
                    psycopg_conn.rollback()
                except Exception:
                    pass

            # Step 2: Create temporary table for this batch
            print("      • Creating temporary table...")
            temp_table_start = time.time()
            cursor.execute("""
                CREATE TEMP TABLE temp_batch_embeddings (
                    song_id INTEGER NOT NULL,
                    embedding vector(2000) NOT NULL
                )
            """)
            # CRITICAL: Commit after CREATE to finish this command
            psycopg_conn.commit()
            print(f"      • Temp table created ({time.time() - temp_table_start:.2f}s)")

            # Step 3: Use COPY to bulk load (NO PRINT STATEMENTS DURING COPY!)
            print(f"      • Copying {len(batch_data):,} rows to temp table...")
            copy_start = time.time()

            # COPY operation - DO NOT PRINT during this operation to avoid "reentrant call" errors
            with cursor.copy(
                "COPY temp_batch_embeddings (song_id, embedding) FROM STDIN"
            ) as copy:
                for song_id, embedding in batch_data:
                    embedding_str = "[" + ",".join(map(str, embedding)) + "]"
                    copy.write_row([int(song_id), embedding_str])

            # CRITICAL: Commit after COPY to finish this command
            psycopg_conn.commit()
            copy_time = time.time() - copy_start
            print(
                f"      • COPY complete ({copy_time:.2f}s, {len(batch_data) / copy_time:.0f} rows/sec)"
            )

            # Step 4: Merge into final table
            print("      • Merging into song_embeddings table...")
            merge_start = time.time()
            cursor.execute("""
                INSERT INTO song_embeddings (song_id, embedding)
                SELECT song_id, embedding
                FROM temp_batch_embeddings
                ON CONFLICT (song_id) 
                DO UPDATE SET 
                    embedding = EXCLUDED.embedding
            """)
            # CRITICAL: Commit after INSERT to finish this command
            psycopg_conn.commit()
            merge_time = time.time() - merge_start
            print(f"      • Merge complete ({merge_time:.2f}s)")

            # Step 5: Clean up temp table
            print("      • Cleaning up temp table...")
            cursor.execute("DROP TABLE IF EXISTS temp_batch_embeddings")
            psycopg_conn.commit()

            # Close cursor
            cursor.close()
            cursor = None

            total_time = time.time() - batch_start_time
            print(
                f"    ✓ Batch complete: {len(batch_data):,} records in {total_time:.2f}s ({len(batch_data) / total_time:.0f} rows/sec)"
            )

            return len(batch_data), 0

        except Exception as e:
            error_type = type(e).__name__
            error_msg = str(e)

            print(f"    ⚠️  COPY batch failed{batch_desc}")
            print(f"        Error type: {error_type}")
            print(f"        Error: {error_msg[:200]}")

            # Cleanup on error
            try:
                if cursor and not cursor.closed:
                    cursor.close()
            except Exception:
                pass

            try:
                psycopg_conn.rollback()
                print("        • Transaction rolled back")
            except Exception as rb_err:
                print(f"        • Rollback error: {rb_err}")

            # Try to clean up temp table
            try:
                cleanup_cursor = psycopg_conn.cursor()
                cleanup_cursor.execute("DROP TABLE IF EXISTS temp_batch_embeddings")
                psycopg_conn.commit()
                cleanup_cursor.close()
                print("        • Temp table cleaned up")
            except Exception:
                pass

            return 0, len(batch_data)

    def _insert_batch(
        sqlalchemy_conn, batch_data: list, batch_desc: str = ""
    ) -> tuple[int, int]:
        """
        Use bulk INSERT statement as fallback. Returns (success_count, error_count).
        Uses SQLAlchemy connection.
        """
        if not batch_data:
            return 0, 0

        batch_start_time = time.time()
        try:
            print(
                f"    → Starting INSERT operation for {len(batch_data):,} records{batch_desc}"
            )

            # Build bulk INSERT statement with multiple VALUES
            print("      • Building SQL statement...")
            build_start = time.time()
            values_list = []
            for idx, (song_id, embedding) in enumerate(batch_data):
                embedding_str = "[" + ",".join(map(str, embedding)) + "]"
                # Escape single quotes in the embedding string
                embedding_str = embedding_str.replace("'", "''")
                values_list.append(f"('{song_id}', CAST('{embedding_str}' AS vector))")

                # Progress for very large batches
                if len(batch_data) > 50000 and (idx + 1) % 25000 == 0:
                    print(
                        f"        ⋯ Built {idx + 1:,}/{len(batch_data):,} VALUES clauses..."
                    )

            values_sql = ",\n".join(values_list)
            build_time = time.time() - build_start
            print(f"      • SQL built ({build_time:.2f}s)")

            insert_sql = f"""
                INSERT INTO song_embeddings (song_id, embedding)
                VALUES {values_sql}
                ON CONFLICT (song_id) 
                DO UPDATE SET 
                    embedding = EXCLUDED.embedding
            """

            print("      • Executing INSERT statement...")
            exec_start = time.time()
            with sqlalchemy_conn.begin():
                sqlalchemy_conn.execute(text(insert_sql))
            exec_time = time.time() - exec_start
            print(f"      • INSERT complete ({exec_time:.2f}s)")

            total_time = time.time() - batch_start_time
            print(
                f"    ✓ Batch complete: {len(batch_data):,} records in {total_time:.2f}s ({len(batch_data) / total_time:.0f} rows/sec)"
            )

            return len(batch_data), 0

        except Exception as e:
            print(
                f"    ⚠️  INSERT batch of {len(batch_data)} failed{batch_desc}: {str(e)[:150]}"
            )
            return 0, len(batch_data)

    def _process_batch_with_fallback(
        conn, batch_data: list, level: int = 0, is_psycopg: bool = False
    ) -> tuple[int, int]:
        """
        Process a batch with fallback strategy.
        Returns (success_count, error_count).
        """
        if _interrupted or not batch_data:
            return 0, 0

        # Define batch sizes for each level
        batch_sizes = [100000, 10000, 1000]

        if level >= len(batch_sizes):
            # At the smallest level, try individual inserts as last resort
            success = 0
            errors = 0

            if is_psycopg:
                # Use psycopg3 for individual inserts
                for song_id, embedding in batch_data:
                    try:
                        embedding_str = "[" + ",".join(map(str, embedding)) + "]"
                        with conn.cursor() as cur:
                            cur.execute(
                                """
                                INSERT INTO song_embeddings 
                                (song_id, embedding)
                                VALUES (%s, %s::vector)
                                ON CONFLICT (song_id) 
                                DO UPDATE SET 
                                    embedding = EXCLUDED.embedding
                            """,
                                (int(song_id), embedding_str),
                            )
                        conn.commit()
                        success += 1
                    except Exception as e:
                        try:
                            conn.rollback()
                        except Exception:
                            pass
                        errors += 1
                        if errors <= 10:
                            print(
                                f"    ⚠️  Error storing embedding for song_id {song_id}: {e}"
                            )
            else:
                # Use SQLAlchemy for individual inserts
                for song_id, embedding in batch_data:
                    try:
                        embedding_str = "[" + ",".join(map(str, embedding)) + "]"
                        with conn.begin():
                            conn.execute(
                                text("""
                                    INSERT INTO song_embeddings 
                                    (song_id, embedding)
                                    VALUES (:song_id, CAST(:embedding AS vector))
                                    ON CONFLICT (song_id) 
                                    DO UPDATE SET 
                                        embedding = EXCLUDED.embedding
                                """),
                                {"song_id": song_id, "embedding": embedding_str},
                            )
                        success += 1
                    except Exception as e:
                        errors += 1
                        if errors <= 10:
                            print(
                                f"    ⚠️  Error storing embedding for song_id {song_id}: {e}"
                            )
            return success, errors

        current_batch_size = batch_sizes[level]
        total_success = 0
        total_errors = 0
        num_chunks = (len(batch_data) + current_batch_size - 1) // current_batch_size

        print(
            f"\n  📦 Processing {len(batch_data):,} records in {num_chunks:,} chunk(s) of {current_batch_size:,} (level {level})"
        )

        # Split data into chunks of current_batch_size
        for chunk_idx, i in enumerate(range(0, len(batch_data), current_batch_size), 1):
            if _interrupted:
                break

            chunk = batch_data[i : i + current_batch_size]
            first_song_id = chunk[0][0]
            last_song_id = chunk[-1][0]

            print(
                f"\n  Chunk {chunk_idx}/{num_chunks}: Processing song_ids {first_song_id:,}-{last_song_id:,}"
            )

            # Use COPY for psycopg3, INSERT for SQLAlchemy
            if is_psycopg:
                success, errors = _copy_batch(
                    conn,
                    chunk,
                    batch_desc=f" (size: {len(chunk):,}, level: {batch_sizes[level]:,})",
                )
            else:
                success, errors = _insert_batch(
                    conn,
                    chunk,
                    batch_desc=f" (size: {len(chunk):,}, level: {batch_sizes[level]:,})",
                )

            if errors > 0:
                # This chunk failed, try with smaller batch size
                next_level_desc = (
                    f"{batch_sizes[level + 1]:,}"
                    if level + 1 < len(batch_sizes)
                    else "individual"
                )
                print(
                    f"    ⚠️  Chunk failed, retrying with smaller batches ({next_level_desc})"
                )
                success, errors = _process_batch_with_fallback(
                    conn, chunk, level + 1, is_psycopg
                )

            total_success += success
            total_errors += errors

            # Progress update
            percent_complete = (total_success / len(batch_data)) * 100
            print(
                f"  ✓ Cumulative progress: {total_success:,}/{len(batch_data):,} embeddings stored ({percent_complete:.1f}%)"
            )
            if total_errors > 0:
                print(f"    Errors so far: {total_errors:,}")

        return total_success, total_errors

    # Convert dataframe to list of tuples for processing
    print("\nPreparing data for upload...")
    convert_start = time.time()
    batch_data = [
        (row["song_id"], row["embedding"])
        for row in df_with_embeddings.iter_rows(named=True)
    ]
    convert_time = time.time() - convert_start
    print(f"✓ Converted {len(batch_data):,} records to tuples ({convert_time:.2f}s)")

    if batch_data:
        print(f"  First song_id: {batch_data[0][0]:,}")
        print(f"  Last song_id: {batch_data[-1][0]:,}")
        print(f"  Embedding dimensions: {len(batch_data[0][1])}")

    # Use appropriate connection type based on availability
    overall_start = time.time()
    if use_copy:
        # Use psycopg3 for COPY method
        print("\n" + "=" * 80)
        print("ESTABLISHING DATABASE CONNECTION (psycopg3 - COPY method)")
        print("=" * 80)
        try:
            print("Connecting to database...")
            conn_start = time.time()
            with psycopg.connect(url, autocommit=False) as conn:
                conn_time = time.time() - conn_start
                print(f"✓ Connected ({conn_time:.2f}s)")

                print("Registering pgvector types...")
                register_vector(conn)
                print("✓ Vector types registered")

                print("\nStarting batch upload process...")
                stored_count, error_count = _process_batch_with_fallback(
                    conn, batch_data, is_psycopg=True
                )
        except Exception as e:
            print(f"\n❌ Error with psycopg3 COPY method: {e}")
            print("   Falling back to SQLAlchemy INSERT method...")
            print("\n" + "=" * 80)
            print("ESTABLISHING DATABASE CONNECTION (SQLAlchemy - INSERT method)")
            print("=" * 80)
            with get_connection(postgres_url) as conn:
                print("✓ Connected")
                print("\nStarting batch upload process...")
                stored_count, error_count = _process_batch_with_fallback(
                    conn, batch_data, is_psycopg=False
                )
    else:
        # Use SQLAlchemy for INSERT method
        print("\n" + "=" * 80)
        print("ESTABLISHING DATABASE CONNECTION (SQLAlchemy - INSERT method)")
        print("=" * 80)
        print("Connecting to database...")
        with get_connection(postgres_url) as conn:
            print("✓ Connected")
            print("\nStarting batch upload process...")
            stored_count, error_count = _process_batch_with_fallback(
                conn, batch_data, is_psycopg=False
            )

    overall_time = time.time() - overall_start
    print("\n" + "=" * 80)
    print(
        f"UPLOAD COMPLETE - Total time: {overall_time:.2f}s ({overall_time / 60:.1f}m)"
    )
    print("=" * 80)

    if _interrupted:
        print(
            f"\n⚠️  Interrupted. Stored {stored_count:,} embeddings before interruption."
        )
    else:
        # Verify the data was stored
        with get_connection(postgres_url) as conn:
            result = conn.execute(text("SELECT COUNT(*) FROM song_embeddings"))
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
                        should_checkpoint = (
                            processed_count % checkpoint_interval == 0
                            or processed_count == total_batches
                        )

                        if should_checkpoint and checkpoint_results:
                            song_ids, embeddings = _extract_embeddings_from_checkpoint(
                                checkpoint_results, df, batch_size
                            )
                            stored = _store_checkpoint(
                                song_ids, embeddings, postgres_url
                            )

                            if stored > 0:
                                total_stored += stored
                                checkpoint_results = []  # Clear after successful store
                                print(
                                    f"  💾 Checkpoint: Stored {total_stored:,} embeddings to database"
                                )

                        # Progress update every 10 batches
                        should_show_progress = (
                            processed_count % 10 == 0
                            or processed_count == total_batches
                        )

                        if should_show_progress:
                            _calculate_and_print_progress(
                                processed_count,
                                total_batches,
                                all_results,
                                total_errors,
                                batch_size,
                                start_time,
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
