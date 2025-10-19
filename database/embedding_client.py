"""
Client for generating embeddings using Fireworks AI API.
"""

import os
from typing import List, Optional
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


class EmbeddingClient:
    """Client for generating embeddings using Fireworks AI."""

    def __init__(self, model: str = "accounts/fireworks/models/qwen3-embedding-8b"):
        """
        Initialize the embedding client.

        Args:
            model: The embedding model to use
        """
        self.model = model
        self.client = OpenAI(
            base_url="https://api.fireworks.ai/inference/v1",
            api_key=os.environ.get("FIREWORKS_API_KEY"),
        )

    def generate_embeddings_batch(
        self, texts: List[str], retry_on_rate_limit: bool = True
    ) -> List[Optional[List[float]]]:
        """
        Generate embeddings for a batch of texts.

        Args:
            texts: List of text strings to generate embeddings for
            retry_on_rate_limit: Whether to retry on rate limit errors

        Returns:
            List of embedding vectors (or None for failed embeddings)
        """
        if not texts:
            return []

        try:
            response = self.client.embeddings.create(
                input=texts,
                model=self.model,
            )

            # Extract embeddings from response
            embeddings = [data.embedding for data in response.data]
            return embeddings

        except Exception as e:
            print(f"Error generating embeddings: {e}")
            # Return None for all texts on error
            return [None] * len(texts)
