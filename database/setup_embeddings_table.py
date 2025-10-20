"""
One-time setup script to create the song_embeddings table.

This creates a fresh song_embeddings table with pgvector support.
Run this once before loading any data.

Usage:
    python database/setup_embeddings_table.py
"""

import sys
from pathlib import Path

# Add project root to path
if __name__ == "__main__":
    project_root = Path(__file__).parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from database.connection import get_connection
from sqlalchemy import text


def create_embeddings_table():
    """Create the song_embeddings table with pgvector support."""
    print("Creating song_embeddings table...")

    with get_connection() as conn:
        # Enable pgvector extension
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))

        # Create the table
        conn.execute(
            text("""
            CREATE TABLE IF NOT EXISTS song_embeddings (
                song_id INTEGER PRIMARY KEY,
                embedding vector(2000) NOT NULL
            )
        """)
        )

        conn.commit()
        print("✓ Table created successfully")


if __name__ == "__main__":
    create_embeddings_table()
