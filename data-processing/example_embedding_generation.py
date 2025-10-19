"""
Example demonstrating how embeddings are generated based on available data.
This script shows the logic without actually calling the API.
"""

from typing import Optional


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


def main():
    """Demonstrate the embedding text generation logic."""
    print("=" * 80)
    print("EMBEDDING TEXT GENERATION EXAMPLES")
    print("=" * 80)

    # Example 1: Song with lyrics (has_lyrics=True, lyrics available)
    print("\n1. Song with lyrics available:")
    print("-" * 80)
    song1_text = create_text_for_embedding(
        song_name="Bohemian Rhapsody",
        band="Queen",
        has_lyrics=True,
        lyrics="Is this the real life?\nIs this just fantasy?\nCaught in a landslide\nNo escape from reality...",
    )
    print(f"has_lyrics: True")
    print(f"lyrics: Available")
    print(f"\nGenerated text for embedding:")
    print(song1_text)

    # Example 2: Song without lyrics (has_lyrics=False)
    print("\n\n2. Song without lyrics (has_lyrics=False):")
    print("-" * 80)
    song2_text = create_text_for_embedding(
        song_name="Clair de Lune", band="Claude Debussy", has_lyrics=False, lyrics=None
    )
    print(f"has_lyrics: False")
    print(f"lyrics: None")
    print(f"\nGenerated text for embedding:")
    print(song2_text)

    # Example 3: Song marked with has_lyrics but lyrics column is None
    print("\n\n3. Song marked with has_lyrics but lyrics unavailable:")
    print("-" * 80)
    song3_text = create_text_for_embedding(
        song_name="Imagine",
        band="John Lennon",
        has_lyrics=True,
        lyrics=None,  # Missing lyrics data
    )
    print(f"has_lyrics: True")
    print(f"lyrics: None (unavailable)")
    print(f"\nGenerated text for embedding:")
    print(song3_text)
    print("\nNote: Falls back to using only song_name and band when lyrics unavailable")

    # Example 4: Song with empty lyrics string
    print("\n\n4. Song with empty lyrics string:")
    print("-" * 80)
    song4_text = create_text_for_embedding(
        song_name="Silent Track",
        band="Unknown Artist",
        has_lyrics=True,
        lyrics="",  # Empty string
    )
    print(f"has_lyrics: True")
    print(f"lyrics: '' (empty string)")
    print(f"\nGenerated text for embedding:")
    print(song4_text)
    print("\nNote: Empty string is falsy, so falls back to song_name and band only")

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("""
The embedding generation logic follows this priority:
1. If has_lyrics is True AND lyrics text is available → Use song_name + band + lyrics
2. Otherwise → Use only song_name + band

This ensures:
- Rich embeddings for songs with lyrics (better semantic matching)
- Fallback behavior for songs without lyrics (still functional)
- Handles edge cases (missing data, empty strings) gracefully
""")


if __name__ == "__main__":
    main()
