"""
Script to generate embeddings for songs with file-based caching.

This script implements a two-phase pipeline:
1. Generate embeddings and cache them to .txt files
2. Batch upload cached embeddings to PostgreSQL database

Features:
- Automatic resume from max cached song_id
- Detailed performance timing to identify bottlenecks
- Graceful interrupt handling (Ctrl+C)
- File-based caching for reliability

For each song, the script embeds: song_name + band + lyrics (if available).

Usage:
    python data-processing/generate_embeddings.py [OPTIONS]

Options:
    --limit=N                Process only N songs
    --cache-dir=PATH         Cache directory (default: ~/Documents/song_embeddings)
    --batch-size=N           Number of songs per API call (max 2048, default 2048)
    --workers=N              Number of parallel workers for API calls (default 8, max 32)
    --upload-batch-size=N    Number of embeddings per DB upload batch (default 500000)
    --upload-only            Skip generation, only upload cached embeddings to database
    --generate-only          Only generate and cache, don't upload to database
    --no-skip-existing       Regenerate embeddings even if cached
    --create-songs-cache     Create parquet cache of songs table, then exit
    --use-songs-cache        Use parquet cache instead of reading from database (faster)
    --songs-cache=PATH       Path to songs parquet cache (default: ~/Documents/song_embeddings/songs_cache.parquet)

Examples:
    # First-time setup: Create songs cache (faster for subsequent runs)
    python data-processing/generate_embeddings.py --create-songs-cache

    # Generate embeddings using parquet cache (10-100x faster song loading)
    python data-processing/generate_embeddings.py --use-songs-cache

    # Generate and cache only (no DB upload)
    python data-processing/generate_embeddings.py --generate-only --use-songs-cache

    # Upload cached embeddings to database
    python data-processing/generate_embeddings.py --upload-only

    # Test with 1000 songs
    python data-processing/generate_embeddings.py --limit=1000

Note:
    - The script automatically resumes from the last cached song_id
    - Press Ctrl+C once to gracefully exit after the current batch
    - All progress is saved to cache files immediately
"""

import sys
import signal
import time
from pathlib import Path

# Add project root to path if running as a script
if __name__ == "__main__":
    project_root = Path(__file__).parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from database.embeddings import (
    read_songs_from_postgres,
    get_max_cached_song_id,
    write_embedding_to_file,
    upload_cached_embeddings_to_database,
    cache_songs_to_parquet,
    load_songs_from_parquet,
    RateLimiter,
)
from database.embedding_client import EmbeddingClient
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock


# Global flag for interrupt handling
_interrupted = False


def signal_handler(signum, frame):
    """Handle keyboard interrupt (Ctrl+C)."""
    global _interrupted
    _interrupted = True
    print(
        "\n\n⚠️  Interrupt received. Finishing current batch and exiting gracefully..."
    )
    print("⚠️  Press Ctrl+C again to force quit")


def format_time(seconds):
    """Format seconds into human-readable time."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds / 60:.1f}m"
    else:
        hours = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        return f"{hours}h {mins}m"


def process_single_batch(
    batch_num: int,
    batch_df,
    cache_dir: str,
    embedding_client: EmbeddingClient,
    rate_limiter: RateLimiter,
    start_from_song_id: int = None,
):
    """
    Process a single batch: generate embeddings and write to cache.

    Args:
        batch_num: Batch number for tracking
        batch_df: Polars DataFrame with songs to process
        cache_dir: Directory to cache embeddings
        embedding_client: Client for generating embeddings
        rate_limiter: Rate limiter for API calls
        start_from_song_id: Offset for progress reporting

    Returns:
        Dictionary with batch statistics and timing
    """
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

    # Generate embeddings (time API call)
    api_start = time.time()
    try:
        # Wait for rate limiter
        rate_limiter.acquire()

        batch_embeddings = embedding_client.generate_embeddings_batch(
            texts_for_embedding
        )

        # Check if all embeddings are None (indicates API failure)
        if batch_embeddings and all(e is None for e in batch_embeddings):
            print(
                f"  ⚠️  Batch {batch_num + 1}: All embeddings failed (API error)",
                flush=True,
            )
    except Exception as e:
        print(f"  ⚠️  Batch {batch_num + 1} unexpected error: {e}", flush=True)
        batch_embeddings = [None] * len(texts_for_embedding)
    api_time = time.time() - api_start

    # Write embeddings to files (time file writes)
    file_write_start = time.time()
    song_ids = batch_df["song_id"].to_list()

    successful = 0
    errors = 0

    for song_id, embedding in zip(song_ids, batch_embeddings):
        if embedding is not None:
            success = write_embedding_to_file(song_id, embedding, cache_dir)
            if success:
                successful += 1
            else:
                errors += 1
        else:
            errors += 1

    file_write_time = time.time() - file_write_start

    return {
        "batch_num": batch_num,
        "song_ids": song_ids,
        "successful": successful,
        "errors": errors,
        "api_time": api_time,
        "file_write_time": file_write_time,
        "total_time": api_time + file_write_time,
    }


def generate_and_cache_embeddings(
    cache_dir: str,
    table_name: str = None,
    limit: int = None,
    batch_size: int = 2048,
    skip_existing: bool = True,
    num_workers: int = 8,
    use_parquet_cache: bool = False,
    parquet_cache_path: str = None,
):
    """
    Phase 1: Generate embeddings and cache them to files.

    Args:
        cache_dir: Directory to store cached embeddings
        table_name: Database table to read from
        limit: Optional limit on number of songs to process
        batch_size: Number of songs per API call (max 2048)
        skip_existing: If True, resume from max cached song_id
        num_workers: Number of parallel workers for API calls (default 8)
        use_parquet_cache: If True, load songs from parquet cache instead of database
        parquet_cache_path: Path to parquet cache file

    Returns:
        Dictionary with statistics
    """
    print("\n" + "=" * 80)
    print("PHASE 1: GENERATE AND CACHE EMBEDDINGS")
    print("=" * 80)

    # Expand cache directory path
    cache_path = Path(cache_dir).expanduser()
    cache_path.mkdir(parents=True, exist_ok=True)
    print(f"Cache directory: {cache_path}")

    # Check for existing cached embeddings
    start_from_song_id = None
    if skip_existing:
        max_cached = get_max_cached_song_id(cache_dir)
        if max_cached is not None:
            start_from_song_id = max_cached
            print(f"✓ Found cached embeddings up to song_id {max_cached}")
            print(f"  Resuming from song_id {max_cached + 1}")
        else:
            print("✓ No cached embeddings found, starting from beginning")
    else:
        print("⚠️  Regenerating all embeddings (--no-skip-existing)")

    # Read songs from database or parquet cache
    print("\n" + "-" * 80)
    if use_parquet_cache:
        print("Reading songs from parquet cache...")
        print("-" * 80)
        df = load_songs_from_parquet(
            parquet_path=parquet_cache_path,
            start_from_song_id=start_from_song_id,
            limit=limit,
        )
    else:
        print("Reading songs from database...")
        print("-" * 80)
        df = read_songs_from_postgres(
            table_name=table_name,
            limit=limit,
            start_from_song_id=start_from_song_id,
        )

    if len(df) == 0:
        print("\n✓ No songs to process. All embeddings are already cached!")
        return {
            "total_songs": 0,
            "embeddings_generated": 0,
            "errors": 0,
            "interrupted": False,
        }

    print(f"\n✓ Will process {len(df):,} songs")

    # Count songs with vs without lyrics (only if not already shown by parquet loader)
    if not use_parquet_cache:
        lyrics_count = df.filter(
            (df["lyrics"].is_not_null()) & (df["lyrics"].str.strip_chars() != "")
        ).shape[0]
        metadata_count = df.shape[0] - lyrics_count
        print(f"  Songs with lyrics: {lyrics_count:,}")
        print(f"  Songs without lyrics (metadata only): {metadata_count:,}")

    # Initialize embedding client and rate limiter
    print("\n" + "-" * 80)
    print("Generating embeddings with parallel processing...")
    print("-" * 80)
    print(f"Batch size: {batch_size} songs per API call")
    print("Model: qwen3-embedding-8b (2000 dimensions)")
    print(f"Parallel workers: {num_workers} concurrent API calls")

    embedding_client = EmbeddingClient()
    rate_limiter = RateLimiter(
        requests_per_minute=5800
    )  # Slightly under 6000 for safety

    # Track statistics
    total_songs = len(df)
    num_batches = (total_songs + batch_size - 1) // batch_size
    print(f"Total batches: {num_batches:,}")
    print("\nProgress reporting:")
    print("  - Update after every completed batch")
    print("  - Detailed report every 10,000 songs")

    # Timing accumulators (thread-safe)
    progress_lock = Lock()
    total_api_time = 0.0
    total_file_write_time = 0.0
    overall_start = time.time()

    successful_count = 0
    error_count = 0
    processed_count = 0

    # Prepare all batch tasks
    batch_tasks = []
    for batch_num in range(num_batches):
        start_idx = batch_num * batch_size
        end_idx = min((batch_num + 1) * batch_size, total_songs)
        batch_df = df.slice(start_idx, end_idx - start_idx)
        batch_tasks.append((batch_num, batch_df))

    # Process batches in parallel
    all_results = [None] * num_batches

    print("\nStarting parallel processing...")
    print(f"Submitting {num_batches:,} batches to {num_workers} workers...\n")

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Submit all tasks
        future_to_batch = {
            executor.submit(
                process_single_batch,
                batch_num,
                batch_df,
                cache_dir,
                embedding_client,
                rate_limiter,
                start_from_song_id,
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
                    successful_count += result["successful"]
                    error_count += result["errors"]
                    total_api_time += result["api_time"]
                    total_file_write_time += result["file_write_time"]
                    processed_count += len(result["song_ids"])

                    # Show brief update after every batch
                    elapsed = time.time() - overall_start
                    rate = processed_count / elapsed if elapsed > 0 else 0
                    print(
                        f"  Batch completed: {processed_count:,}/{total_songs:,} songs ({rate:.0f}/sec, {successful_count:,} cached, {error_count} errors)",
                        flush=True,
                    )

                    # Detailed report every 10,000 songs
                    if processed_count % 10000 == 0 or processed_count == total_songs:
                        # Calculate batch timing for this 10K chunk
                        batch_start_song = max(1, processed_count - 10000 + 1)
                        if start_from_song_id:
                            batch_start_song += start_from_song_id
                        batch_end_song = processed_count
                        if start_from_song_id:
                            batch_end_song += start_from_song_id

                        first_song_id = result["song_ids"][0]
                        last_song_id = result["song_ids"][-1]

                        print(f"\n{'=' * 80}")
                        print(
                            f"Progress Report (songs {batch_start_song:,}-{batch_end_song:,}):"
                        )
                        print(f"{'=' * 80}")
                        print(
                            f"  Last completed batch song_ids: {first_song_id:,} - {last_song_id:,}"
                        )
                        print(f"  Rate: {rate:.1f} songs/sec")
                        print("  ")
                        print("Cumulative stats:")
                        print(
                            f"  Total processed: {processed_count:,} / {total_songs:,}"
                        )
                        print(f"  Successful: {successful_count:,}")
                        print(f"  Errors: {error_count:,}")
                        print(f"  Total API time: {format_time(total_api_time)}")
                        print(
                            f"  Total file write time: {format_time(total_file_write_time)}"
                        )
                        print(f"  Elapsed: {format_time(elapsed)}")

                        # Calculate ETA
                        if rate > 0:
                            remaining = total_songs - processed_count
                            eta_seconds = remaining / rate
                            print(f"  ETA: {format_time(eta_seconds)}")

                        print(f"{'=' * 80}")

            except Exception as e:
                print(f"  ⚠️  Batch {batch_num + 1} unexpected error: {e}")
                with progress_lock:
                    error_count += batch_size

    total_elapsed = time.time() - overall_start

    print("\n" + "=" * 80)
    print("PHASE 1 COMPLETE")
    print("=" * 80)
    print(f"Total songs processed: {processed_count:,}")
    print(f"Embeddings cached: {successful_count:,}")
    print(f"Errors: {error_count:,}")
    print(f"Total time: {format_time(total_elapsed)}")
    print(f"Average rate: {processed_count / total_elapsed:.1f} songs/sec")
    print("")
    print("Performance breakdown:")
    print(
        f"  API time: {format_time(total_api_time)} ({total_api_time / total_elapsed * 100:.1f}%)"
    )
    print(
        f"  File write time: {format_time(total_file_write_time)} ({total_file_write_time / total_elapsed * 100:.1f}%)"
    )
    print(
        f"  Other overhead: {format_time(total_elapsed - total_api_time - total_file_write_time)}"
    )
    print("=" * 80)

    return {
        "total_songs": processed_count,
        "embeddings_generated": successful_count,
        "errors": error_count,
        "interrupted": _interrupted,
    }


def main():
    """Main entry point for the script."""
    global _interrupted

    # Register signal handler for Ctrl+C
    signal.signal(signal.SIGINT, signal_handler)

    # Parse command line arguments
    limit = None
    table_name = None
    cache_dir = "~/Documents/song_embeddings"
    batch_size = 2048
    upload_batch_size = 500000
    num_workers = 8
    skip_existing = True
    mode = "both"  # both, generate-only, upload-only, create-songs-cache
    use_parquet_cache = False
    parquet_cache_path = "~/Documents/song_embeddings/songs_cache.parquet"

    for arg in sys.argv[1:]:
        if arg.startswith("--limit="):
            try:
                limit = int(arg.split("=")[1])
            except ValueError:
                print(f"Invalid limit value: {arg}")
                sys.exit(1)
        elif arg.startswith("--table="):
            table_name = arg.split("=")[1]
        elif arg.startswith("--cache-dir="):
            cache_dir = arg.split("=")[1]
        elif arg.startswith("--batch-size="):
            try:
                batch_size = int(arg.split("=")[1])
                if batch_size < 1 or batch_size > 2048:
                    print("Batch size must be between 1 and 2048")
                    sys.exit(1)
            except ValueError:
                print(f"Invalid batch size value: {arg}")
                sys.exit(1)
        elif arg.startswith("--upload-batch-size="):
            try:
                upload_batch_size = int(arg.split("=")[1])
                if upload_batch_size < 1:
                    print("Upload batch size must be at least 1")
                    sys.exit(1)
            except ValueError:
                print(f"Invalid upload batch size value: {arg}")
                sys.exit(1)
        elif arg.startswith("--workers="):
            try:
                num_workers = int(arg.split("=")[1])
                if num_workers < 1 or num_workers > 32:
                    print("Number of workers must be between 1 and 32")
                    sys.exit(1)
            except ValueError:
                print(f"Invalid workers value: {arg}")
                sys.exit(1)
        elif arg == "--no-skip-existing":
            skip_existing = False
        elif arg == "--generate-only":
            mode = "generate-only"
        elif arg == "--upload-only":
            mode = "upload-only"
        elif arg == "--create-songs-cache":
            mode = "create-songs-cache"
        elif arg == "--use-songs-cache":
            use_parquet_cache = True
        elif arg.startswith("--songs-cache="):
            parquet_cache_path = arg.split("=")[1]
            use_parquet_cache = True  # Automatically enable if path is specified
        elif arg in ["--help", "-h"]:
            print(__doc__)
            sys.exit(0)

    # Handle create-songs-cache mode first
    if mode == "create-songs-cache":
        print("=" * 80)
        print("CREATE SONGS PARQUET CACHE")
        print("=" * 80)
        print(f"Source table: {table_name or 'songs'}")
        print(f"Cache file: {parquet_cache_path}")
        print("=" * 80)

        try:
            count = cache_songs_to_parquet(
                parquet_path=parquet_cache_path,
                table_name=table_name,
            )
            print(f"\n✓ Successfully cached {count:,} songs to parquet")
            print("\nNext steps:")
            print("  1. Use --use-songs-cache to load songs from cache")
            print("  2. This will be 10-100x faster than reading from database")
            sys.exit(0)
        except Exception as e:
            print(f"\n❌ Error creating songs cache: {e}")
            import traceback

            traceback.print_exc()
            sys.exit(1)

    # Print configuration for other modes
    print("=" * 80)
    print("EMBEDDING GENERATION WITH FILE CACHING")
    print("=" * 80)
    print(f"Mode: {mode}")
    print(f"Cache directory: {cache_dir}")
    print(f"Source table: {table_name or 'songs'}")
    if use_parquet_cache:
        print(f"Songs source: Parquet cache ({parquet_cache_path})")
    else:
        print("Songs source: Database")

    if mode != "upload-only":
        print(f"API batch size: {batch_size} songs per call")
        print(f"Parallel workers: {num_workers}")
        if limit:
            print(f"Limit: {limit} songs")
        else:
            print("Limit: None (process all)")
        print(f"Auto-resume: {'Enabled' if skip_existing else 'Disabled'}")

    if mode != "generate-only":
        print(f"DB upload batch size: {upload_batch_size:,} embeddings per batch")

    print("=" * 80)

    try:
        phase1_stats = None
        phase2_uploaded = None

        # Phase 1: Generate and cache
        if mode in ["both", "generate-only"]:
            phase1_stats = generate_and_cache_embeddings(
                cache_dir=cache_dir,
                table_name=table_name,
                limit=limit,
                batch_size=batch_size,
                skip_existing=skip_existing,
                num_workers=num_workers,
                use_parquet_cache=use_parquet_cache,
                parquet_cache_path=parquet_cache_path,
            )

            if phase1_stats["interrupted"]:
                print("\n⚠️  Generation phase was interrupted")
                if mode == "both":
                    print(
                        "⚠️  Skipping upload phase. Run with --upload-only to upload cached embeddings."
                    )
                sys.exit(130)

        # Phase 2: Upload to database
        if mode in ["both", "upload-only"]:
            if not _interrupted:
                phase2_uploaded = upload_cached_embeddings_to_database(
                    cache_dir=cache_dir,
                    batch_size=upload_batch_size,
                )

        # Print final summary
        print("\n" + "=" * 80)
        print("FINAL SUMMARY")
        print("=" * 80)

        if phase1_stats:
            print("Phase 1 (Generation):")
            print(f"  Songs processed: {phase1_stats['total_songs']:,}")
            print(f"  Embeddings cached: {phase1_stats['embeddings_generated']:,}")
            print(f"  Errors: {phase1_stats['errors']:,}")

        if phase2_uploaded is not None:
            print("\nPhase 2 (Upload):")
            print(f"  Embeddings uploaded to database: {phase2_uploaded:,}")

        print("=" * 80)
        print("✓ PIPELINE COMPLETE")
        print("=" * 80)

    except KeyboardInterrupt:
        print("\n\n⚠️  Force quit detected. Exiting immediately.")
        sys.exit(130)
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
    finally:
        # Restore default signal handler
        signal.signal(signal.SIGINT, signal.SIG_DFL)


if __name__ == "__main__":
    main()
