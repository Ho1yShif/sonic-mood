# Database

This directory contains database utilities and scripts for managing PostgreSQL connections, embedding tables, and vector similarity search.

## Overview

The database package provides:
- Connection management with optimized pooling strategies for different use cases
- Embedding table setup and index creation
- Embedding client for generating vectors via Fireworks AI
- High-performance batch operations for storing embeddings
- Cache management utilities for embedding generation pipelines

## Scripts

Database scripts integrate with the data-processing workflow. See [data processing documentation](../data-processing/README.md) for details.

### `connection.py`
Database connection utilities with SQLAlchemy engine management.

**Features:**
- Three connection pooling strategies:
  - **Single-threaded batch**: Uses `NullPool` for simple scripts
  - **Web server**: Optimized for concurrent web requests (pool_size=5, max_overflow=10)
  - **Parallel batch**: Optimized for parallel processing (pool_size=15 for 8 workers)
- Connection keepalive for reliable long-running processes
- Automatic pgvector extension initialization

**Usage:**
```python
from database.connection import get_connection, get_engine

# Context manager for automatic cleanup
with get_connection(for_web_server=True) as conn:
    result = conn.execute(text("SELECT * FROM songs"))

# Or get engine directly
engine = get_engine(for_parallel_batch=True)
```

### `embedding_client.py`
Client for generating embeddings using the Fireworks AI API.

**Features:**
- Batch embedding generation (up to 2048 texts per request)
- Uses `qwen3-embedding-8b` model with 2000 dimensions
- Automatic error handling and rate limit detection
- OpenAI-compatible client interface

**Usage:**
```python
from database.embedding_client import EmbeddingClient

client = EmbeddingClient()
texts = ["Song name by Artist", "Another song by Another artist"]
embeddings = client.generate_embeddings_batch(texts)
```

### `embeddings.py`
Comprehensive embedding operations and cache management.

**Features:**
- **Parquet caching**: Cache songs data to parquet for 10-100x faster loading
- **File-based caching**: Write embeddings to disk for reliable resume
- **Bulk upload**: High-performance COPY-based database inserts (10-100x faster than individual INSERTs)
- **Rate limiting**: Thread-safe token bucket rate limiter for API calls
- **Progress tracking**: Detailed progress reports for long-running operations
- **Graceful interrupts**: Ctrl+C handling with checkpoint saves

**Key functions:**

#### Cache management
- `cache_songs_to_parquet()`: Cache all songs from DB to parquet file
- `load_songs_from_parquet()`: Load songs from parquet cache
- `get_max_cached_song_id()`: Find max cached song_id for resume
- `write_embedding_to_file()`: Write embedding to cache file
- `read_embedding_from_file()`: Read embedding from cache file
- `load_embeddings_from_cache()`: Load multiple embeddings from cache

#### Database operations
- `read_songs_from_postgres()`: Read songs with lyrics from database
- `upload_cached_embeddings_to_database()`: Batch upload cached embeddings
- `store_embeddings_copy_bulk()`: Fast COPY-based bulk insert (recommended)
- `store_embeddings_batch()`: Fallback batch insert method

#### Rate limiting
- `RateLimiter`: Thread-safe rate limiter class for API throttling

## Setup scripts

### `setup_embeddings_table.py`
One-time setup script to create the `song_embeddings` table with pgvector support.

**What it does:**
- Enables pgvector extension
- Creates `song_embeddings` table with:
  - `song_id` (INTEGER PRIMARY KEY)
  - `embedding` (vector(8192) NOT NULL)
  - `created_at` (TIMESTAMP)

**Usage:**
```bash
python database/setup_embeddings_table.py
```

Run this once before generating and storing embeddings.

### `create_index.py`
Creates an IVFFlat index on the `song_embeddings` table for efficient vector similarity search.

**What it does:**
- Creates IVFFlat index using `vector_cosine_ops`
- Uses `lists = 3525` (optimized for ~12M rows, based on sqrt(rows) recommendation)
- Index name: `embeddings_ivfflat_idx`

**Usage:**
```bash
python database/create_index.py
```

**Important:** Run this AFTER all embeddings have been loaded into the table. Index creation may take several minutes depending on data size.

## Environment variables

Required environment variables (use `.env` file):
- `POSTGRES_URL`: PostgreSQL connection string (format: `postgresql://user:password@host:port/database`)
- `FIREWORKS_API_KEY`: API key for Fireworks AI (for embedding generation)
