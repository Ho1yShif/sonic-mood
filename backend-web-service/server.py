import os

from openai import OpenAI
from sanic import Sanic
from sanic.log import logger
from sanic.request import Request
from sanic.response import json
from sanic_cors import CORS
from dotenv import load_dotenv
from spotipy import Spotify
from spotipy.oauth2 import SpotifyClientCredentials

from semantic_search import Song, search_database

load_dotenv()


app = Sanic("InferenceBackend")
CORS(app)

client = OpenAI(
    base_url="https://api.fireworks.ai/inference/v1",
    api_key=os.environ.get("FIREWORKS_API_KEY"),
)

# Initialize Spotify client (cached at module level - only initialized once)
spotify_client_id = os.environ.get("SPOTIFY_CLIENT_ID")
spotify_client_secret = os.environ.get("SPOTIFY_CLIENT_SECRET")

# Debug logging to see what we're getting
logger.info(f"Spotify Client ID present: {bool(spotify_client_id)}")
logger.info(f"Spotify Client Secret present: {bool(spotify_client_secret)}")

if spotify_client_id and spotify_client_secret:
    try:
        auth_manager = SpotifyClientCredentials(
            client_id=spotify_client_id, client_secret=spotify_client_secret
        )
        spotify = Spotify(auth_manager=auth_manager)
        logger.info("✓ Spotify client successfully initialized")
    except Exception as e:
        logger.error(f"Failed to initialize Spotify client: {e}")
        spotify = None
else:
    logger.warning(
        "Spotify credentials not found. Spotify links will not be available."
    )
    logger.warning(f"SPOTIFY_CLIENT_ID is {'set' if spotify_client_id else 'NOT set'}")
    logger.warning(
        f"SPOTIFY_CLIENT_SECRET is {'set' if spotify_client_secret else 'NOT set'}"
    )
    spotify = None

# Cache for Spotify links to avoid redundant API calls
# Key: (title, artist) tuple, Value: Spotify link (or None if not found)
spotify_link_cache: dict[tuple[str, str], str | None] = {}


async def embed(text: str) -> list[float]:
    response = client.embeddings.create(
        input=text,
        model="accounts/fireworks/models/qwen3-embedding-8b",
        dimensions=2000,
    )

    return response.data[0].embedding


async def get_hypothetical_song(query: str) -> str:
    response = client.chat.completions.create(
        model="accounts/fireworks/models/qwen3-235b-a22b-instruct-2507",
        max_tokens=512,
        temperature=1.0,
        messages=[
            {
                "role": "system",
                "content": """\
You are a creative musician and songwriter. When a user sends you a message, write a song for them that captures the vibe they're looking for.
Start by writing the song title, then the song lyrics. Use only plain text, no markdown formatting. Respond only with the song, do not say anything to the user.
""",
            },
            {
                "role": "user",
                "content": f"""\
Write a song that matches this query:
{query}

Respond only with the song, no other text.""",
            },
        ],
    )

    model_answer = response.choices[0].message.content.split("</think>")[-1].strip()
    logger.debug(f"HyDE model response: {model_answer}")
    return model_answer


async def get_playlist_name(songs: list[Song], query: str) -> str:
    formatted_songs = "\n- ".join(
        [f"{song['title']} by {song['artist']}" for song in songs]
    )

    response = client.chat.completions.create(
        model="accounts/fireworks/models/qwen3-235b-a22b-instruct-2507",
        max_tokens=512,
        temperature=1.1,
        messages=[
            {
                "role": "system",
                "content": "You are a creative and witty music playlist curator. For a given list of songs, you are tasked with coming up with a three to four word playlist name that captures the mood and genres. (eg: Experimental Rock Hypnosis, Guilty Pleasure Sleepover, or Panicked Girl Dinner).",
            },
            {
                "role": "user",
                "content": f"""\
I came up with this playlist of songs based on the prompt '{query}':
{formatted_songs}

Please write a name for the playlist.
Respond only with the name of the playlist, no other text.""",
            },
        ],
    )

    model_answer = response.choices[0].message.content.split("</think>")[-1].strip()

    logger.debug(f"Playlist name model response: {model_answer}")
    return model_answer


async def add_spotify_links(songs: list[Song]) -> list[Song]:
    """
    Search for each song on Spotify and add the Spotify link if found.
    Uses caching to avoid redundant API calls for the same song.
    """
    if not spotify:
        logger.warning("Spotify client not initialized. Skipping Spotify links.")
        return songs

    for song in songs:
        cache_key = (song["title"], song["artist"])

        # Check if we already have the link cached
        if cache_key in spotify_link_cache:
            song["spotifyLink"] = spotify_link_cache[cache_key]
            logger.debug(
                f"Using cached Spotify link for {song['title']} by {song['artist']}"
            )
            continue

        # Not in cache, fetch from Spotify API
        try:
            # Search for the song on Spotify using title and artist
            query = f"track:{song['title']} artist:{song['artist']}"
            results = spotify.search(q=query, type="track", limit=1)

            # Check if we found any tracks
            if results["tracks"]["items"]:
                track = results["tracks"]["items"][0]
                song["spotifyLink"] = track["external_urls"]["spotify"]
                logger.debug(
                    f"Found Spotify link for {song['title']} by {song['artist']}: {song['spotifyLink']}"
                )
            else:
                logger.debug(
                    f"No Spotify link found for {song['title']} by {song['artist']}"
                )
                song["spotifyLink"] = None

            # Cache the result (whether we found a link or not)
            spotify_link_cache[cache_key] = song["spotifyLink"]

        except Exception as e:
            logger.error(
                f"Error searching Spotify for {song['title']} by {song['artist']}: {e}"
            )
            song["spotifyLink"] = None
            # Cache the None result to avoid retrying failed lookups
            spotify_link_cache[cache_key] = None

    return songs


@app.post("/recommendations")
async def get_recommendations(request: Request) -> dict:
    query = request.json.get("query")
    if not query:
        return json({"error": "Query is required"}, status=400)

    logger.info(f"Generating playlist for '{query}'...")

    # Step 1: Generate a hypothetical song that will help us match similar songs
    logger.info("Writing a hypothetical song...")
    hypothetical_song = await get_hypothetical_song(query)

    # Step 2: Create the vector embeddings
    logger.info("Computing vector embeddings...")
    query_embedding = await embed(query)
    hyde_embedding = await embed(hypothetical_song)

    # Step 3: Use the vector embeddings to search the database for similar songs
    logger.info("Searching for similar songs...")
    query_songs = await search_database(query_embedding, n=5)
    hyde_songs = await search_database(hyde_embedding, n=3)
    songs = query_songs + hyde_songs
    # Remove duplicates by title and artist
    songs = list({(song["title"], song["artist"]): song for song in songs}.values())

    # Step 4: Generate a title for the given songs
    logger.info("Coming up with clever playlist title...")
    playlist_name = await get_playlist_name(songs=songs, query=query)

    # Step 5: Add the Spotify links to the songs
    logger.info("Getting Spotify links for the songs...")
    songs = await add_spotify_links(songs=songs)

    logger.info(f"Done! Created playlist '{playlist_name}'.")
    return json(
        {
            "playlist_name": playlist_name,
            "songs": songs,
        }
    )
