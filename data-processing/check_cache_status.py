#!/usr/bin/env python3
"""
Quick diagnostic script to check embedding cache status.
"""

import sys
from pathlib import Path
import time

# Add project root to path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from database.embeddings import get_max_cached_song_id


def main():
    cache_dir = "~/Documents/song_embeddings"
    cache_path = Path(cache_dir).expanduser()

    print("=" * 80)
    print("EMBEDDING CACHE STATUS")
    print("=" * 80)
    print(f"Cache directory: {cache_path}")
    print()

    if not cache_path.exists():
        print("❌ Cache directory does not exist!")
        return

    # Count cache files
    cache_files = list(cache_path.glob("*.txt"))
    print(f"Total cache files: {len(cache_files):,}")

    if not cache_files:
        print("❌ No cache files found!")
        return

    # Get max cached song_id
    max_song_id = get_max_cached_song_id(cache_dir)
    print(f"Max cached song_id: {max_song_id:,}")

    # Get the 10 most recently modified files
    cache_files_sorted = sorted(
        cache_files, key=lambda p: p.stat().st_mtime, reverse=True
    )

    print("\nMost recently modified cache files:")
    current_time = time.time()
    for i, file_path in enumerate(cache_files_sorted[:10], 1):
        song_id = file_path.stem
        mod_time = file_path.stat().st_mtime
        age_seconds = current_time - mod_time
        age_minutes = age_seconds / 60

        if age_minutes < 1:
            age_str = f"{age_seconds:.0f}s ago"
        elif age_minutes < 60:
            age_str = f"{age_minutes:.1f}m ago"
        else:
            age_hours = age_minutes / 60
            age_str = f"{age_hours:.1f}h ago"

        print(f"  {i}. song_id {song_id}.txt - modified {age_str}")

    # Check if files are still being created
    most_recent = cache_files_sorted[0]
    most_recent_age = current_time - most_recent.stat().st_mtime

    print()
    print("=" * 80)
    if most_recent_age < 60:
        print("✓ Cache files are being actively created (most recent < 1 min ago)")
    elif most_recent_age < 300:
        print("⚠️  Cache files might be stalled (most recent > 1 min ago)")
    else:
        print(
            f"❌ Cache files appear stalled (most recent {most_recent_age / 60:.1f} min ago)"
        )
        print("   The program may have encountered errors. Check terminal output for:")
        print("   - '❌ API ERROR' messages")
        print("   - Rate limit warnings")
        print("   - Network errors")
    print("=" * 80)


if __name__ == "__main__":
    main()
