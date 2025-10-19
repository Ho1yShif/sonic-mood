# Sonic mood

Sonic mood showcases how to use [Render](https://render.com) to deploy an AI-powered music recommendation app that leverages [pgvector](https://github.com/pgvector/pgvector) for semantic search.

## Quickstart

## Architecture

- Song, interaction, and lyrics data along with their vector embeddings live in a Render-managed PostgreSQL database equipped with pgvector
- The backend web service performs the following functions:
  - Embeds user queries using [`qwen3-embedding-8b`](https://huggingface.co/Qwen/Qwen3-Embedding-8B) via [Fireworks](https://fireworks.ai/)
  - Creates hypothetical song lyrics from the query via [`qwen3-235b-a22b-instruct-2507`](https://huggingface.co/Qwen/Qwen3-235B-A22B-Instruct-2507) to better match the lyrics embeddings with [HyDE techniques](https://milvus.io/ai-quick-reference/what-is-hyde-hypothetical-document-embeddings-and-when-should-i-use-it)
  - Connects to the PostgreSQL database and uses pgvector to search the PostgreSQL database for songs that are similar to the user's query
  - Outputs a playlist of songs that are semantically similar to the user's query
  - Generates a short, fun playlist title to summarize its mood in the same vein as a Spotify daylist via [`qwen3-235b-a22b-instruct-2507`](https://huggingface.co/Qwen/Qwen3-235B-A22B-Instruct-2507)
  - Retrieves Spotify links to output songs when available with caching
- The static site displays output from the backend in a delightful, clean UI
- Render deployments:
  - PostgreSQL database
  - Backend web service
  - Static site frontend

### Design decisions

### Why firework

### Why Qwen models?

#### Why HyDE?

## Style
- Clean, one-page design
- Glassmorphic top bar
- Modern search bar with `Cmd+K` functionality
- Purple accents and highlights to match the standout Render color
- Highlights on example queries reminscent of the responsive effects in docs
- Custom logos created in Canva
- <img src="frontend-site/src/assets/note_purple.png" alt="custom" style={{width: '50%'}} />
- <img src="frontend-site/src/assets/note_white.png" alt="custom" style={{width: '50%'}} />

## Data processing steps
The following are one-time data processing steps performed in sequence:
1. Joined `track_meta.tsv` and `user_item_interaction.csv` data from [Kaggle dataset](https://www.kaggle.com/datasets/usasha/million-music-playlists/data). The resulting `songs` dataset contained 12,431,127 rows
1. Removed `37` rows with a null `song_name`, leaving 12,431,090 rows remaining
1. Enriched `songs` with interaction stats from `user_item_interaction.csv` including `interactions_count`, `unique_users`,`avg_interactions_per_user`, and `popularity_score` (`0.6 * interactions_count + 0.4 * unique_users`)
1. Enriched `songs` with lyrics data to produce `enriched_songs`
   a. After attempting to use `lyricsgenius` and hitting API rate limits that would prove impossible with my allotted time frame, I elected to join the `songs` data to an [existing Genius dataset](https://huggingface.co/datasets/sebastiandizon/genius-song-lyrics) rather than collecting song lyrics from the API myself
   b. Only 128,667 songs (~1%) could be joined to the lyrics dataset even with normalization
1. Stored `enriched_songs` in the Postgres database
1. Generated vector embeddings for each song using `song_name`, `band`, and `lyrics` (if available) fields to capture the strongest possible semantic information

