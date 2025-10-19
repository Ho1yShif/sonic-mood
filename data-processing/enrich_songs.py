import os
import polars as pl
import kagglehub
from typing import Optional
from datasets import load_dataset

# Load Kaggle data
dataset_path = kagglehub.dataset_download("usasha/million-music-playlists")
tracks = pl.scan_csv(os.path.join(dataset_path, "track_meta.tsv"), separator="\t")
interactions = pl.scan_csv(
    os.path.join(dataset_path, "user_item_interaction.csv"),
    schema_overrides={"user": pl.String},
)


def load_genius_lyrics_dataset(
    sample_size: Optional[int] = None, streaming: bool = False
) -> pl.DataFrame:
    """
    Load the Genius Song Lyrics dataset from Hugging Face.

    Args:
        sample_size: Optional number of rows to load (useful for testing)
        streaming: If True, use streaming mode for memory efficiency

    Returns:
        Polars DataFrame with lyrics data
    """
    print("Loading Genius Song Lyrics dataset from Hugging Face...")
    print("Dataset: sebastiandizon/genius-song-lyrics")

    if streaming:
        print("Loading in streaming mode...")
        dataset = load_dataset(
            "sebastiandizon/genius-song-lyrics", split="train", streaming=True
        )
        if sample_size:
            print(f"Taking first {sample_size:,} rows...")
            dataset = dataset.take(sample_size)

        # Convert to list of dicts then to Polars
        data = list(dataset)
        df = pl.DataFrame(data)
    else:
        print("Loading full dataset...")
        dataset = load_dataset("sebastiandizon/genius-song-lyrics", split="train")
        df = pl.DataFrame(dataset.to_pandas())
        print(f"Dataset loaded: {len(df):,} total rows available")

        if sample_size:
            print(f"Sampling {sample_size:,} rows...")
            df = df.sample(n=min(sample_size, len(df)))

    print(f"Loaded {len(df):,} songs from Hugging Face")
    print(f"Columns: {df.columns}")

    return df


def normalize_text(text: str) -> str:
    """
    Normalize text for matching (lowercase, remove extra spaces, etc.).
    This is a simple version that can be enhanced for better matching.
    """
    import re
    import string

    if not text:
        return ""
    # Lowercase, remove all punctuation, normalize spaces
    text = text.lower().strip()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text)
    return text


def compute_interaction_stats(interactions_df: pl.DataFrame) -> pl.DataFrame:
    """
    Compute interaction statistics per song.

    Args:
        interactions_df: DataFrame with 'item' (song_id) and 'user' columns

    Returns:
        DataFrame with song interaction statistics:
        - interactions_count: total number of interactions
        - unique_users: number of unique users who interacted
        - avg_interactions_per_user: average interactions per user
        - popularity_score: weighted score (0.6 * interactions_count + 0.4 * unique_users)
    """
    stats = (
        interactions_df.group_by("item")
        .agg(
            [
                pl.len().alias("interactions_count"),
                pl.col("user").n_unique().alias("unique_users"),
            ]
        )
        .with_columns(
            [
                (pl.col("interactions_count") / pl.col("unique_users")).alias(
                    "avg_interactions_per_user"
                ),
                (
                    0.6 * pl.col("interactions_count") + 0.4 * pl.col("unique_users")
                ).alias("popularity_score"),
            ]
        )
    )
    return stats


def join_tracks_and_interactions() -> pl.DataFrame:
    """
    Load tracks and enrich with interaction statistics at the song level.

    Returns:
        DataFrame with one row per song, including track metadata and interaction stats
    """
    print("Loading tracks dataset...")
    tracks_df = tracks.collect()

    # Filter out songs with null song_name
    original_count = len(tracks_df)
    tracks_df = tracks_df.filter(pl.col("song_name").is_not_null())
    filtered_count = len(tracks_df)
    print(f"Filtered out {original_count - filtered_count:,} songs with null song_name")
    print(f"Remaining tracks: {filtered_count:,}")

    print("Loading interactions dataset...")
    interactions_df = interactions.collect()

    print("\nComputing interaction statistics per song...")
    interaction_stats = compute_interaction_stats(interactions_df)

    print(f"\n{'=' * 80}")
    print("Interaction Statistics (top 10 by popularity_score):")
    print(f"{'=' * 80}")
    print(interaction_stats.sort("popularity_score", descending=True).head(10))
    print(f"{'=' * 80}")

    print("\nJoining tracks with interaction statistics...")
    # Join tracks with interaction stats (song-level data)
    joined_df = tracks_df.join(
        interaction_stats, left_on="song_id", right_on="item", how="left"
    )

    # Display results
    print(f"\n{'=' * 80}")
    print("Song-Level Dataset with Interaction Stats (first 25 rows):")
    print(f"{'=' * 80}")
    print(joined_df.head(25))
    print(f"\n{'=' * 80}")
    print(f"Shape: {joined_df.shape} | Columns: {joined_df.columns}")

    return joined_df


def join_lyrics_with_songs(
    songs_df: pl.DataFrame, lyrics_df: pl.DataFrame, match_method: str = "exact"
) -> pl.DataFrame:
    """
    Join the Genius lyrics dataset with our songs data.

    Args:
        songs_df: Our songs DataFrame (with song_name, band columns)
        lyrics_df: Genius lyrics DataFrame (with title, artist columns)
        match_method: "exact" for exact matching, "normalized" for fuzzy matching

    Returns:
        Joined DataFrame with lyrics added
    """
    print("\nJoining lyrics with songs data...")
    print(f"Songs to match: {len(songs_df):,}")
    print(f"Lyrics available: {len(lyrics_df):,}")

    # Select relevant columns from lyrics dataset
    lyrics_subset = lyrics_df.select(
        [
            pl.col("title").alias("lyrics_title"),
            pl.col("artist").alias("lyrics_artist"),
            "lyrics",
        ]
    )

    if match_method == "normalized":
        print("Using normalized matching (more flexible)...")
        # Create normalized columns for matching
        songs_with_norm = songs_df.with_columns(
            [
                pl.col("song_name")
                .map_elements(normalize_text)
                .alias("song_name_norm"),
                pl.col("band").map_elements(normalize_text).alias("band_norm"),
            ]
        )

        lyrics_with_norm = lyrics_subset.with_columns(
            [
                pl.col("lyrics_title").map_elements(normalize_text).alias("title_norm"),
                pl.col("lyrics_artist")
                .map_elements(normalize_text)
                .alias("artist_norm"),
            ]
        )

        # Join on normalized names
        joined = songs_with_norm.join(
            lyrics_with_norm,
            left_on=["song_name_norm", "band_norm"],
            right_on=["title_norm", "artist_norm"],
            how="left",
        ).drop(["song_name_norm", "band_norm"])

    else:  # exact matching
        print("Using exact matching...")
        joined = songs_df.join(
            lyrics_subset,
            left_on=["song_name", "band"],
            right_on=["lyrics_title", "lyrics_artist"],
            how="left",
        )

    # Add has_lyrics boolean column
    joined = joined.with_columns(pl.col("lyrics").is_not_null().alias("has_lyrics"))

    # Count matches
    matched_count = joined.filter(pl.col("lyrics").is_not_null()).height
    match_rate = (matched_count / len(songs_df)) * 100 if len(songs_df) > 0 else 0

    print(f"\n{'=' * 80}")
    print("Match Results:")
    print(f"  Total songs: {len(songs_df):,}")
    print(f"  Matched with lyrics: {matched_count:,} ({match_rate:.1f}%)")
    print(f"  No lyrics found: {len(songs_df) - matched_count:,}")
    print(f"{'=' * 80}")

    return joined


def enrich_songs_with_lyrics(
    songs_df: pl.DataFrame,
    sample_lyrics: Optional[int] = None,
    match_method: str = "exact",
) -> pl.DataFrame:
    """
    Main function to enrich songs with lyrics from Hugging Face dataset.

    Args:
        songs_df: Our songs DataFrame
        sample_lyrics: Optional limit on lyrics dataset size (for testing)
        match_method: "exact" or "normalized" matching

    Returns:
        Enriched DataFrame with lyrics (songs without lyrics are filtered out)
    """
    # Load lyrics dataset
    lyrics_df = load_genius_lyrics_dataset()

    # Join with our songs
    enriched_df = join_lyrics_with_songs(songs_df, lyrics_df, match_method)

    # Show sample results
    print(f"\n{'=' * 80}")
    print("Sample of enriched data:")
    print(f"{'=' * 80}")
    if len(enriched_df) > 0:
        sample = enriched_df.head(5)
        print(sample)
    else:
        print("No matches found!")
    print(f"{'=' * 80}\n")

    return enriched_df


if __name__ == "__main__":
    # Example usage:
    joined_df = join_tracks_and_interactions()
    enriched_df = enrich_songs_with_lyrics(joined_df)
    print(enriched_df.head())
