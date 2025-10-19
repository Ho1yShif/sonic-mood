"""
Quick test script to verify database connectivity and pgvector queries work correctly.
"""

import asyncio
import os
from dotenv import load_dotenv

load_dotenv()


async def test_database_connection():
    """Test that we can connect to the database and query song_embeddings."""
    from semantic_search import db_engine, search_database
    from sqlalchemy import text

    if not db_engine:
        print(
            "❌ ERROR: Database not initialized. Check POSTGRES_URL environment variable."
        )
        return False

    print("✓ Database engine initialized")

    try:
        # Test 1: Check if song_embeddings table exists
        print("\n1. Checking if song_embeddings table exists...")
        with db_engine.connect() as conn:
            result = conn.execute(
                text(
                    """
                SELECT COUNT(*) FROM song_embeddings
            """
                )
            )
            count = result.fetchone()[0]
            print(f"✓ Found {count:,} embeddings in song_embeddings table")

            if count == 0:
                print(
                    "⚠️  WARNING: song_embeddings table is empty. Run generate_embeddings.py first."
                )
                return False

        # Test 2: Get a sample embedding from the database
        print("\n2. Fetching a sample embedding...")
        with db_engine.connect() as conn:
            result = conn.execute(
                text(
                    """
                SELECT e.song_id, e.embedding, s.song_name, s.band
                FROM song_embeddings e
                JOIN songs s ON e.song_id = s.song_id
                LIMIT 1
            """
                )
            )
            row = result.fetchone()
            if row:
                sample_song_id = row[0]
                sample_embedding = row[1]
                sample_name = row[2]
                sample_artist = row[3]
                print(f"✓ Sample song: '{sample_name}' by {sample_artist}")
                print(f"  Song ID: {sample_song_id}")
                print(f"  Embedding dimension: {len(sample_embedding)}")
            else:
                print("❌ ERROR: Could not fetch sample embedding")
                return False

        # Test 3: Test similarity search with the sample embedding
        print("\n3. Testing similarity search (should return the same song first)...")
        try:
            songs = await search_database(
                sample_embedding, n=3, use_hybrid_scoring=False
            )
            if songs:
                print(f"✓ Search returned {len(songs)} songs:")
                for i, song in enumerate(songs, 1):
                    print(f"  {i}. {song['title']} by {song['artist']}")

                # The first result should be the same song we queried with
                if (
                    songs[0]["title"] == sample_name
                    and songs[0]["artist"] == sample_artist
                ):
                    print(
                        "\n✓ SUCCESS: First result matches input song (perfect similarity)"
                    )
                else:
                    print("\n⚠️  WARNING: First result doesn't match input song")
                    print(f"   Expected: '{sample_name}' by {sample_artist}")
                    print(f"   Got: '{songs[0]['title']}' by {songs[0]['artist']}")
            else:
                print("❌ ERROR: Search returned no results")
                return False
        except Exception as e:
            print(f"❌ ERROR during search: {e}")
            import traceback

            traceback.print_exc()
            return False

        # Test 4: Test hybrid scoring
        print("\n4. Testing hybrid scoring...")
        try:
            songs_hybrid = await search_database(
                sample_embedding, n=3, use_hybrid_scoring=True, popularity_weight=0.3
            )
            if songs_hybrid:
                print(f"✓ Hybrid search returned {len(songs_hybrid)} songs:")
                for i, song in enumerate(songs_hybrid, 1):
                    print(f"  {i}. {song['title']} by {song['artist']}")
            else:
                print("❌ ERROR: Hybrid search returned no results")
                return False
        except Exception as e:
            print(f"❌ ERROR during hybrid search: {e}")
            import traceback

            traceback.print_exc()
            return False

        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED!")
        print("=" * 60)
        print("\nYour database setup is working correctly!")
        return True

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(test_database_connection())
    exit(0 if success else 1)
