"""Gemini embedding helpers for the FastAPI backend."""
import os, time
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

EMBEDDING_MODEL = "gemini-embedding-001"
OUTPUT_DIM      = 768
BATCH_SIZE      = 20
MAX_RETRIES     = 6


def _client() -> genai.Client:
    return genai.Client(api_key=os.environ["GEMINI_API_KEY"])


def _embed_batch(texts: list[str]) -> list[list[float]]:
    client = _client()
    for attempt in range(MAX_RETRIES):
        try:
            response = client.models.embed_content(
                model=EMBEDDING_MODEL,
                contents=texts,
                config=types.EmbedContentConfig(output_dimensionality=OUTPUT_DIM),
            )
            return [list(e.values) for e in response.embeddings]
        except Exception as e:
            err = str(e).lower()
            wait = 10 * (2 ** attempt)
            if "429" in err or "quota" in err or "rate" in err or "resource" in err:
                time.sleep(wait)
            else:
                raise
    raise RuntimeError("Embedding failed after max retries.")


def embed_texts(texts: list[str]) -> list[list[float]]:
    all_vectors: list[list[float]] = []
    for i in range(0, len(texts), BATCH_SIZE):
        all_vectors.extend(_embed_batch(texts[i: i + BATCH_SIZE]))
    return all_vectors


def embed_single(text: str) -> list[float]:
    return _embed_batch([text])[0]
