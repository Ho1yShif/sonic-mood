"""
Script to generate embeddings for songs and store them in PostgreSQL.
This script reads songs from the database, generates embeddings, and stores them in
the separate song_embeddings table.

The script automatically uses lyrics if the lyrics column is populated, otherwise
it falls back to metadata (song_name + band).

Usage:
    python data-processing/generate_embeddings.py [OPTIONS]

Options:
    --limit=N           Process only N songs
    --table=NAME        Read from specific table (default: songs)
    --no-skip-existing  Process all songs, even if they already have embeddings
    --batch-size=N      Number of songs per API call (max 2048, default 2048)

Examples:
    # Generate embeddings for 100 songs
    python data-processing/generate_embeddings.py --limit=100

    # Generate embeddings for all songs
    python data-processing/generate_embeddings.py

    # Regenerate embeddings (don't skip existing)
    python data-processing/generate_embeddings.py --no-skip-existing

    # Use smaller batch size (e.g., for testing)
    python data-processing/generate_embeddings.py --batch-size=100

Note: Press Ctrl+C once to gracefully exit after the current batch.
      Press Ctrl+C twice to force quit (may lose progress).
"""

import sys
from pathlib import Path

# Add project root to path if running as a script
if __name__ == "__main__":
    project_root = Path(__file__).parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from database.embeddings import generate_and_store_all_embeddings


def main():
    """Main entry point for the script."""
    # Parse command line arguments
    limit = None
    table_name = None  # Will default to songs
    skip_existing = True
    batch_size = 2048  # Default batch size (max allowed by API)

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
        elif arg == "--no-skip-existing":
            skip_existing = False
        elif arg in ["--help", "-h"]:
            print(__doc__)
            sys.exit(0)

    print(f"Source table: {table_name or 'songs'}")
    print("Target table: song_embeddings")
    print("Mode: Will use lyrics if populated, otherwise metadata (song_name + band)")
    print(f"Batch size: {batch_size} songs per API call")
    if limit:
        print(f"Processing {limit} songs...")
    else:
        print("Processing all songs...")

    if skip_existing:
        print(
            "Skipping songs with existing embeddings (note: check disabled when auto-detecting types)"
        )
    else:
        print("Processing all songs (will update existing embeddings)")

    print("\nPress Ctrl+C once to gracefully exit after the current batch.")
    print("Press Ctrl+C twice to force quit (may lose progress).\n")

    try:
        # Run the embedding generation pipeline
        stats = generate_and_store_all_embeddings(
            table_name=table_name,
            limit=limit,
            skip_existing=skip_existing,
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
            sys.exit(130)  # Standard exit code for SIGINT

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
