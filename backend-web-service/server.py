import os
from typing import TypedDict

from openai import OpenAI
from sanic import Sanic
from sanic.log import logger
from sanic.request import Request
from sanic.response import json
from sanic_cors import CORS
from dotenv import load_dotenv

load_dotenv()


app = Sanic("InferenceBackend")
CORS(app)

client = OpenAI(
    base_url="https://api.fireworks.ai/inference/v1",
    api_key=os.environ.get("FIREWORKS_API_KEY"),
)


class Song(TypedDict):
    title: str
    artist: str


async def embed(text: str) -> list[float]:
    response = client.embeddings.create(
        input=text,
        model="accounts/fireworks/models/qwen3-embedding-8b",
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


async def search_database(query_embedding: list[float], n: int = 5) -> list[Song]:
    # TODO: Use pgvector to search the PostgreSQL database for similar songs
    return [
        {"title": "Inside My Head", "artist": "Sincere Engineer"},
        {"title": "Awkward", "artist": "Weakened Friends"},
        {"title": "Less Happy More Free", "artist": "Ben Lapidus"},
        {"title": "I Think You Should Leave", "artist": "Winona Fighter"},
        {"title": "EVERYTHING CAUSES CANCER", "artist": "Gavin Prophet"},
        {"title": "I Need You to Kill Me", "artist": "Ditzy Spells"},
        {"title": "Not Everyone is Gonna Love You", "artist": "Mattstagraham"},
    ][:n]


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

    # Step 4: Generate a title for the given songs
    logger.info("Coming up with clever playlist title...")
    playlist_name = await get_playlist_name(songs=songs, query=query)

    logger.info(f"Done! Created playlist '{playlist_name}'.")
    return json(
        {
            "playlist_name": playlist_name,
            "songs": songs,
        }
    )
