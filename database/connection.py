"""
Database connection utilities.
"""

import os
from contextlib import contextmanager
from typing import Generator, Optional
from sqlalchemy import create_engine, Engine, Connection, text
from sqlalchemy.pool import NullPool
from dotenv import load_dotenv

load_dotenv()


_engine = None
_web_engine = None
_batch_parallel_engine = None


def get_engine(
    postgres_url: str = None,
    for_web_server: bool = False,
    for_parallel_batch: bool = False,
) -> Engine:
    """
    Get or create a SQLAlchemy engine.

    Args:
        postgres_url: PostgreSQL connection URL (defaults to POSTGRES_URL env var)
        for_web_server: If True, use pooling optimized for concurrent web requests.
        for_parallel_batch: If True, use pooling optimized for parallel batch processing.
                           If both False, use NullPool for single-threaded batch operations.

    Returns:
        SQLAlchemy Engine instance
    """
    global _engine, _web_engine, _batch_parallel_engine

    url = postgres_url or os.environ.get("POSTGRES_URL")

    if for_web_server:
        if _web_engine is None:
            _web_engine = create_engine(
                url,
                pool_pre_ping=True,
                pool_size=5,
                max_overflow=10,
                pool_recycle=3600,
                connect_args={
                    "connect_timeout": 10,
                    "keepalives": 1,
                    "keepalives_idle": 30,
                    "keepalives_interval": 10,
                    "keepalives_count": 5,
                },
            )
        return _web_engine
    elif for_parallel_batch:
        if _batch_parallel_engine is None:
            _batch_parallel_engine = create_engine(
                url,
                pool_pre_ping=True,
                pool_size=15,  # Support 8 workers + some overhead
                max_overflow=5,
                pool_recycle=3600,
                connect_args={
                    "connect_timeout": 10,
                },
            )
        return _batch_parallel_engine
    else:
        if _engine is None:
            _engine = create_engine(
                url,
                poolclass=NullPool,  # Use NullPool for single-threaded batch operations
            )
        return _engine


@contextmanager
def get_connection(
    postgres_url: str = None,
    for_web_server: bool = False,
    for_parallel_batch: bool = False,
) -> Generator[Connection, None, None]:
    """
    Context manager for database connections.

    Args:
        postgres_url: PostgreSQL connection URL (defaults to POSTGRES_URL env var)
        for_web_server: If True, use pooling optimized for concurrent web requests
        for_parallel_batch: If True, use pooling optimized for parallel batch processing

    Yields:
        SQLAlchemy Connection instance

    Example:
        with get_connection() as conn:
            result = conn.execute(text("SELECT * FROM songs"))
    """
    engine = get_engine(
        postgres_url,
        for_web_server=for_web_server,
        for_parallel_batch=for_parallel_batch,
    )
    conn = engine.connect()
    try:
        yield conn
    finally:
        conn.close()


def initialize_pgvector(postgres_url: str = None, for_web_server: bool = False) -> bool:
    """
    Initialize pgvector extension in the database.

    Args:
        postgres_url: PostgreSQL connection URL (defaults to POSTGRES_URL env var)
        for_web_server: If True, use web server engine

    Returns:
        True if successful, False otherwise
    """
    try:
        with get_connection(postgres_url, for_web_server=for_web_server) as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            conn.commit()
        return True
    except Exception as e:
        print(f"Failed to enable pgvector extension: {e}")
        return False


def dispose_engine():
    """Dispose of the current engines and reset them."""
    global _engine, _web_engine, _batch_parallel_engine
    if _engine:
        _engine.dispose()
        _engine = None
    if _web_engine:
        _web_engine.dispose()
        _web_engine = None
    if _batch_parallel_engine:
        _batch_parallel_engine.dispose()
        _batch_parallel_engine = None
