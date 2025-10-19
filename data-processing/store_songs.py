"""
Store enriched music data into PostgreSQL database.
Uses SQLAlchemy for reliable connections with proper SSL handling.
"""

import os
import time
import polars as pl
from sqlalchemy import (
    create_engine,
    text,
    Table,
    Column,
    Integer,
    String,
    Float,
    Boolean,
    MetaData,
)
from sqlalchemy.exc import OperationalError, DatabaseError
from enrich_songs import join_tracks_and_interactions, enrich_songs_with_lyrics


def retry_with_backoff(func, max_retries=5, initial_delay=1):
    """Retry a function with exponential backoff."""
    delay = initial_delay
    last_exception = None

    for attempt in range(max_retries):
        try:
            return func()
        except (OperationalError, DatabaseError) as e:
            last_exception = e
            if attempt < max_retries - 1:
                print(
                    f"  Connection error (attempt {attempt + 1}/{max_retries}): {str(e)[:100]}"
                )
                print(f"  Retrying in {delay} seconds...")
                time.sleep(delay)
                delay *= 2
            else:
                print(f"  Failed after {max_retries} attempts")
                raise last_exception


def setup_table(engine, songs_table, table_name, if_table_exists):
    """Setup database table based on if_table_exists mode."""
    with engine.begin() as conn:
        if if_table_exists == "replace":
            print("  ⚠️  WARNING: Dropping existing table (if exists) and recreating...")
            songs_table.drop(conn, checkfirst=True)
            songs_table.create(conn)
        elif if_table_exists == "fail":
            songs_table.create(conn)
        else:  # append or upsert
            songs_table.create(conn, checkfirst=True)


def get_existing_count(engine, table_name):
    """Get count of existing rows in table."""
    with engine.connect() as conn:
        try:
            result = conn.execute(text(f"SELECT COUNT(*) FROM {table_name}"))
            return result.fetchone()[0]
        except Exception:
            return 0


def insert_batch_sqlalchemy(
    engine, table_name, batch, columns_to_store, songs_table, if_table_exists
):
    """
    Insert batch using SQLAlchemy (reliable, handles SSL properly).
    Uses efficient bulk operations with proper conflict handling.
    """
    with engine.begin() as conn:
        if if_table_exists in ("append", "upsert"):
            # Build INSERT with ON CONFLICT clause for safe operations
            columns_list = ", ".join(columns_to_store)
            placeholders = ", ".join([f":{col}" for col in columns_to_store])

            if if_table_exists == "append":
                # Skip duplicates (DO NOTHING on conflict)
                conflict_action = "DO NOTHING"
            else:  # upsert
                # Update existing records
                update_cols = [col for col in columns_to_store if col != "song_id"]
                update_set = ", ".join(
                    [f"{col} = EXCLUDED.{col}" for col in update_cols]
                )
                conflict_action = f"DO UPDATE SET {update_set}"

            insert_sql = text(f"""
                INSERT INTO {table_name} ({columns_list})
                VALUES ({placeholders})
                ON CONFLICT (song_id) {conflict_action}
            """)

            for record in batch:
                conn.execute(insert_sql, record)
        else:
            # Standard bulk insert for 'replace' and 'fail' modes (faster)
            conn.execute(songs_table.insert(), batch)

    return len(batch)


def verify_count(engine, table_name):
    """Verify final row count in table."""
    with engine.connect() as conn:
        result = conn.execute(text(f"SELECT COUNT(*) FROM {table_name}"))
        return result.fetchone()[0]


def store_songs_to_postgres(
    df: pl.DataFrame,
    postgres_url: str,
    table_name: str = "songs",
    if_table_exists: str = "replace",
    min_interactions: int = 2,
    batch_size: int = 5000,
) -> None:
    """
    Store songs data into PostgreSQL database using SQLAlchemy bulk operations.

    This implementation uses SQLAlchemy's connection pooling and transaction
    management, which properly handles SSL connections to Render PostgreSQL.

    Args:
        df: Polars DataFrame containing songs data
        postgres_url: PostgreSQL connection URL
        table_name: Name of the table to create/update
        if_table_exists: 'fail', 'replace', 'append', or 'upsert'
        min_interactions: Minimum number of interactions to include song
        batch_size: Number of rows per batch (5000 is a good balance)
    """
    valid_modes = ["fail", "replace", "append", "upsert"]
    if if_table_exists not in valid_modes:
        raise ValueError(
            f"Invalid if_table_exists: '{if_table_exists}'. Must be one of: {', '.join(valid_modes)}"
        )

    print("\nConnecting to PostgreSQL database...")
    engine = create_engine(
        postgres_url,
        pool_pre_ping=True,
        pool_recycle=3600,
        connect_args={
            "connect_timeout": 10,
            "keepalives": 1,
            "keepalives_idle": 30,
            "keepalives_interval": 10,
            "keepalives_count": 5,
        },
    )

    # Define table schema
    metadata = MetaData()
    columns = [
        Column("song_id", Integer, primary_key=True),
        Column("song_name", String),
        Column("band", String),
        Column("interactions_count", Integer),
        Column("unique_users", Integer),
        Column("avg_interactions_per_user", Float),
        Column("popularity_score", Float),
        Column("has_lyrics", Boolean),
        Column("lyrics", String),
    ]

    columns_to_store = [
        "song_id",
        "song_name",
        "band",
        "interactions_count",
        "unique_users",
        "avg_interactions_per_user",
        "popularity_score",
        "has_lyrics",
        "lyrics",
    ]

    songs_table = Table(table_name, metadata, *columns)
    df_to_store = df.select(columns_to_store)
    total_records = len(df_to_store)

    print(f"Storing {total_records:,} songs into table '{table_name}'...")
    print("Using SQLAlchemy bulk operations with connection pooling")

    # Setup table
    retry_with_backoff(
        lambda: setup_table(engine, songs_table, table_name, if_table_exists)
    )
    print("✓ Table setup complete")

    # Check existing rows
    existing_rows = retry_with_backoff(lambda: get_existing_count(engine, table_name))
    if existing_rows > 0 and if_table_exists in ("append", "upsert"):
        mode_msg = (
            "append (skipping duplicates)"
            if if_table_exists == "append"
            else "upsert (update/insert)"
        )
        print(f"Found {existing_rows:,} existing rows, will {mode_msg}")

    if total_records == 0:
        print("No records to insert")
        engine.dispose()
        return

    # Convert to list of dicts for SQLAlchemy
    records = df_to_store.to_dicts()

    # Start bulk insert operation
    print("Starting bulk insert operation...")
    start_time = time.time()
    rows_processed = 0

    # Process in batches with progress tracking
    for i in range(0, len(records), batch_size):
        batch = records[i : i + batch_size]

        # Insert batch with retry logic
        retry_with_backoff(
            lambda b=batch: insert_batch_sqlalchemy(
                engine, table_name, b, columns_to_store, songs_table, if_table_exists
            )
        )
        rows_processed += len(batch)

        # Progress update
        progress_pct = (rows_processed / total_records) * 100
        elapsed = time.time() - start_time
        rate = rows_processed / elapsed if elapsed > 0 else 0
        print(
            f"  Progress: {rows_processed:,}/{total_records:,} rows ({progress_pct:.1f}%) - {rate:,.0f} rows/sec"
        )

    # Verify final count
    print("\nVerifying final row count...")
    final_count = retry_with_backoff(lambda: verify_count(engine, table_name))

    total_time = time.time() - start_time
    avg_rate = final_count / total_time if total_time > 0 else 0
    print(
        f"✓ Successfully stored {final_count:,} songs in {total_time:.1f}s ({avg_rate:,.0f} rows/sec)"
    )

    engine.dispose()


if __name__ == "__main__":
    POSTGRES_URL = os.environ.get("POSTGRES_URL")

    # Process and enrich songs data with lyrics
    print("=" * 80)
    print("Processing songs data with lyrics...")
    print("=" * 80)
    joined_df = join_tracks_and_interactions()
    enriched_df = enrich_songs_with_lyrics(joined_df)

    # Store complete dataframe with lyrics
    store_songs_to_postgres(
        df=enriched_df,
        postgres_url=POSTGRES_URL,
        table_name="songs",
        if_table_exists="replace",  # Drop and recreate table with fresh schema
        batch_size=5000,  # Balanced batch size for reliable bulk insert
    )
