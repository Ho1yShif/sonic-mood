# Backend Web Service

A Sanic-based API service that powers music recommendations using AI embeddings and Hypothetical Document Embeddings (HyDE) technique.

## Overview

This backend service provides intelligent music recommendations by:

1. Generating hypothetical songs based on user queries using LLMs
2. Creating vector embeddings for semantic search
3. Searching a PostgreSQL database with pgvector for similar songs
4. Generating creative playlist names

## Features

- **HyDE (Hypothetical Document Embeddings)**: Generates a hypothetical song that matches the user's query to improve search relevance
- **Vector Embeddings**: Uses Qwen3 embedding model for semantic similarity
- **Creative Playlist Naming**: Generates witty, mood-capturing playlist names using LLMs
- **RESTful API**: Simple JSON-based API for easy integration

## Tech Stack

- **Web Framework**: Sanic (async Python web framework)
- **AI/ML**: Fireworks AI API (Qwen3 models)
- **Database**: PostgreSQL with pgvector (Render)

## API Endpoints

### POST /recommendations

Generates music recommendations based on a text query.

**Request Body:**
```json
{
  "query": "songs for a rainy day"
}
```

**Response:**
```json
{
  "playlist_name": "Melancholic Rain Vibes",
  "songs": [
    {
      "title": "Inside My Head",
      "artist": "Sincere Engineer"
    },
    {
      "title": "Awkward",
      "artist": "Weakened Friends"
    }
    // ... more songs
  ]
}
```

## How It Works

1. **Hypothetical Song Generation**: The service uses an LLM to write a hypothetical song that matches the user's query
2. **Embedding Creation**: Both the original query and hypothetical song are converted to vector embeddings
3. **Database Search**: The embeddings are used to search the database for similar real songs
4. **Playlist Naming**: An LLM generates a creative 3-4 word playlist name based on the songs and query

## Setup

### Prerequisites

- Python 3.8+
- PostgreSQL with pgvector extension (for production)
- Fireworks AI API key

### Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up environment variables:
Create a `.env` file in the `backend-web-service` directory:
```env
FIREWORKS_API_KEY=your_api_key_here
```

3. Run the server:
```bash
sanic server
```

The server will start on the default Sanic port (typically 8000).

## Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `FIREWORKS_API_KEY` | API key for Fireworks AI services | Yes |

## Models Used

- **Embedding Model**: `qwen3-embedding-8b` - Generates vector embeddings for semantic search
- **LLM Model**: `qwen3-235b-a22b-instruct-2507` - MoE LLM for text generation

## Deployment

This service is designed to be deployed on Render alongside a PostgreSQL database with the pgvector extension.
