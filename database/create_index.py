"""
Create ivfflat index on song_embeddings table.

This should be run AFTER data has been loaded into the table.
The index improves query performance for vector similarity searches.

Usage:
    python database/create_index.py
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


def create_ivfflat_index():
    """Create ivfflat index for efficient similarity search."""
    print("Creating ivfflat index on song_embeddings...")
    print("(This may take a few minutes depending on data size)")

    with get_connection() as conn:
        # Create the index
        conn.execute(
            text("""
            CREATE INDEX embeddings_ivfflat_idx
            ON song_embeddings 
            USING ivfflat (embedding vector_cosine_ops) 
            WITH (lists = 100)
        """)
        )

        conn.commit()
        print("✓ Index created successfully")


if __name__ == "__main__":
    create_ivfflat_index()
