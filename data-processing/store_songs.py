"""
Store enriched music data into PostgreSQL database.
"""

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
    MetaData,
)
from sqlalchemy.exc import OperationalError, DatabaseError
from enrich_songs import join_tracks_and_interactions, enrich_songs_with_lyrics


def retry_with_backoff(func, max_retries=5, initial_delay=1):
    """
    Retry a function with exponential backoff.

    Args:
        func: Function to retry
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay in seconds

    Returns:
        Result of the function
    """
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
                delay *= 2  # Exponential backoff
            else:
                print(f"  Failed after {max_retries} attempts")
                raise last_exception


def store_songs_to_postgres(
    df: pl.DataFrame,
    postgres_url: str,
    table_name: str = "songs",
    if_table_exists: str = "replace",
    min_interactions: int = 2,
    batch_size: int = 5000,
) -> None:
    """
    Store songs data into PostgreSQL database using SQLAlchemy with retry logic.

    Args:
        df: Polars DataFrame containing songs data
        postgres_url: PostgreSQL connection URL
        table_name: Name of the table to create/update
        if_table_exists: What to do if table exists:
            - 'fail': Raise error if table exists
            - 'replace': Drop and recreate table (CAUTION: deletes all existing data)
            - 'append': Append data, skip duplicates on conflict
            - 'upsert': Insert new records and update existing ones
        min_interactions: Minimum number of interactions to include song (filters out noise)
        batch_size: Number of rows per batch (smaller = more resilient)
    """
    # Validate if_table_exists parameter
    valid_modes = ["fail", "replace", "append", "upsert"]
    if if_table_exists not in valid_modes:
        raise ValueError(
            f"Invalid if_table_exists value: '{if_table_exists}'. "
            f"Must be one of: {', '.join(valid_modes)}"
        )

    print("\nConnecting to PostgreSQL database...")
    # Add connection pooling settings for better reliability
    engine = create_engine(
        postgres_url,
        pool_pre_ping=True,  # Verify connections before using
        pool_recycle=3600,  # Recycle connections after 1 hour
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

    # Base columns that are always present
    columns = [
        Column("song_id", Integer, primary_key=True),
        Column("song_name", String),
        Column("band", String),
        Column("interactions_count", Integer),
        Column("unique_users", Integer),
        Column("avg_interactions_per_user", Float),
        Column("popularity_score", Float),
    ]

    columns_to_store = [
        "song_id",
        "song_name",
        "band",
        "interactions_count",
        "unique_users",
        "avg_interactions_per_user",
        "popularity_score",
    ]

    # Optional columns - add if present in DataFrame
    optional_columns = {
        "lyrics": String,
    }

    for col_name, col_type in optional_columns.items():
        if col_name in df.columns:
            columns.append(Column(col_name, col_type))
            columns_to_store.append(col_name)

    songs_table = Table(table_name, metadata, *columns)
    df_to_store = df.select(columns_to_store)

    # Convert polars dataframe to list of dictionaries for bulk insert
    records = df_to_store.to_dicts()
    total_records = len(records)

    print(f"Storing {total_records:,} songs into table '{table_name}'...")

    # Check if we need to create/replace the table
    def setup_table():
        with engine.begin() as conn:
            if if_table_exists == "replace":
                # WARNING: This drops the entire table and all data
                print(
                    "  ⚠️  WARNING: Dropping existing table (if exists) and recreating..."
                )
                songs_table.drop(conn, checkfirst=True)
                songs_table.create(conn)
            elif if_table_exists == "fail":
                songs_table.create(conn)
            elif if_table_exists in ("append", "upsert"):
                # Create table only if it doesn't exist (safe for re-runs)
                songs_table.create(conn, checkfirst=True)

    retry_with_backoff(setup_table)
    print("✓ Table setup complete")

    # Check if we're resuming from a previous run
    def get_existing_count():
        with engine.connect() as conn:
            try:
                result = conn.execute(text(f"SELECT COUNT(*) FROM {table_name}"))
                return result.fetchone()[0]
            except Exception:
                return 0

    existing_rows = retry_with_backoff(get_existing_count)
    if existing_rows > 0:
        if if_table_exists == "append":
            print(
                f"Found {existing_rows:,} existing rows, will append (skipping duplicates)"
            )
        elif if_table_exists == "upsert":
            print(
                f"Found {existing_rows:,} existing rows, will upsert (update existing, insert new)"
            )

    # Bulk insert the data in batches with progress tracking
    if records:
        rows_inserted = existing_rows
        start_idx = 0

        for i in range(start_idx, total_records, batch_size):
            batch = records[i : i + batch_size]

            # Insert batch with retry logic - handle conflicts based on mode
            def insert_batch():
                with engine.begin() as conn:
                    if if_table_exists in ("append", "upsert"):
                        # Build INSERT with ON CONFLICT clause for safe overwrites
                        columns_list = ", ".join(columns_to_store)
                        placeholders = ", ".join(
                            [f":{col}" for col in columns_to_store]
                        )

                        if if_table_exists == "append":
                            # Skip duplicates (DO NOTHING on conflict)
                            conflict_action = "DO NOTHING"
                        else:  # upsert
                            # Update existing records
                            update_cols = [
                                col for col in columns_to_store if col != "song_id"
                            ]
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
                        # Standard insert for 'replace' and 'fail' modes
                        conn.execute(songs_table.insert(), batch)

            retry_with_backoff(insert_batch)
            rows_inserted += len(batch)

            # Show progress after each batch
            print(
                f"  Progress: {rows_inserted:,}/{total_records:,} rows inserted ({rows_inserted / total_records * 100:.1f}%)"
            )

        print(f"  Completed: {rows_inserted:,} rows inserted")

    # Verify the data was stored
    def verify_count():
        with engine.connect() as conn:
            query = text(f"SELECT COUNT(*) FROM {table_name}")
            result = conn.execute(query)
            return result.fetchone()[0]

    count = retry_with_backoff(verify_count)
    print(f"Successfully stored {count:,} songs in PostgreSQL table '{table_name}'")

    engine.dispose()


if __name__ == "__main__":
    POSTGRES_URL = "postgresql://ho1yshif:K2ytIVfh9qNu2Ig6ARnhIxWL6iRlHrnw@dpg-d3pj9lt6ubrc73f7fh20-a.oregon-postgres.render.com/render_take_home"

    # Get the main data (without lyrics)
    print("=" * 80)
    print("Processing main songs data (without lyrics)...")
    print("=" * 80)
    joined_df = join_tracks_and_interactions()

    # Store main songs data with smaller batch size for reliability
    # Use "upsert" for safe re-runs (updates existing, inserts new)
    # Use "replace" for clean slate (drops entire table - CAUTION!)
    store_songs_to_postgres(
        df=joined_df,
        postgres_url=POSTGRES_URL,
        table_name="songs",
        if_table_exists="upsert",  # Changed from "replace" for safer re-runs
        batch_size=5000,
    )

    # Enrich with lyrics sample
    print("\n" + "=" * 80)
    print("Processing enriched songs data (with lyrics)...")
    print("=" * 80)
    enriched_df = enrich_songs_with_lyrics(joined_df)

    # Store enriched songs with lyrics
    store_songs_to_postgres(
        df=enriched_df,
        postgres_url=POSTGRES_URL,
        table_name="songs_with_lyrics",
        if_table_exists="upsert",  # Changed from "replace" for safer re-runs
        batch_size=1000,  # Even smaller for lyrics data
    )
