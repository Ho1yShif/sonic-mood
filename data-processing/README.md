# Data Processing

This directory contains scripts for offline processing to build and enrich the database with song data.

## Scripts

### `enrich_songs.py`
Enriches song data with lyrics from the Genius Song Lyrics dataset on Hugging Face.

**Features:**
- Loads track metadata and user interaction statistics
- Joins with lyrics dataset using exact or normalized matching
- Adds a `has_lyrics` boolean flag to indicate whether lyrics are available

### `store_songs.py`
Stores enriched song data into PostgreSQL database.

**Schema includes:**
- `song_id` (Primary Key)
- `song_name`
- `band`
- `interactions_count`
- `unique_users`
- `avg_interactions_per_user`
- `popularity_score`
- `has_lyrics` (Boolean)
- `lyrics` (Text, nullable)

### `generate_embeddings.py`
Generates vector embeddings for songs and stores them in a separate `song_embeddings` table.

**Embedding Generation Logic:**
- When `has_lyrics` is `True` AND `lyrics` column is available:
  - Uses: `song_name`, `band`, and `lyrics` for embedding
  - Format: `"{song_name} by {band}\n\n{lyrics}"`
  
- When `has_lyrics` is `False` OR `lyrics` column is unavailable:
  - Uses: Only `song_name` and `band` for embedding
  - Format: `"{song_name} by {band}"`

**Usage:**
```bash
# Generate embeddings for all songs
python generate_embeddings.py

# Process only 100 songs
python generate_embeddings.py --limit=100

# Use specific table and batch size
python generate_embeddings.py --table=songs --batch-size=50
```

**Options:**
- `--limit=N`: Process only N songs
- `--table=NAME`: Read from specific table (default: 'songs')
- `--batch-size=N`: Number of songs per API call (max 2048, default 100)
- `--help` or `-h`: Show help message

**Output:**
Creates a `song_embeddings` table with:
- `song_id` (Primary Key, references songs table)
- `embedding` (Array of floats - vector embedding)
- `text_used` (Text snippet showing what was embedded)

### `language.py`
Utility script for detecting and analyzing language distribution in song names.

## Requirements

Install dependencies:
```bash
pip install -r requirements.txt
```

## Workflow

1. **Enrich songs with lyrics:**
   ```bash
   python enrich_songs.py
   ```

2. **Store songs in database:**
   ```bash
   python store_songs.py
   ```

3. **Generate embeddings:**
   ```bash
   python generate_embeddings.py
   ```

## Environment Variables

Required environment variables (use `.env` file):
- `FIREWORKS_API_KEY`: API key for Fireworks AI (for embeddings)
- `POSTGRES_URL`: PostgreSQL connection string (optional, has default)