"""
Phase 4: Embedding generation using Google Gemini text-embedding-001.

Provides:
    embed_chunks()    — embed a list of chunk dicts, returns them with embeddings attached
    embed_sentences() — embed individual sentences (for semantic chunking)
    embed_query()     — embed a single query string for retrieval

Model: gemini-embedding-001 (3072-dim)
Batching: 20 texts per API call (Gemini batch limit)
Retry: exponential backoff on rate limit errors, max 6 retries.
"""

import os
import time
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

EMBEDDING_MODEL = "gemini-embedding-001"
EMBEDDING_DIM   = 768
OUTPUT_DIM      = 768  # truncated from 3072 — fits within pgvector HNSW 2000-dim limit
BATCH_SIZE      = 20
MAX_RETRIES     = 6


def _get_client() -> genai.Client:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError("GEMINI_API_KEY is not set. Add it to your .env file.")
    return genai.Client(api_key=api_key)


def _embed_batch(client: genai.Client, texts: list[str]) -> list[list[float]]:
    """
    Embed a batch of texts with exponential backoff on rate limit errors.
    Returns a list of embedding vectors aligned with input texts.
    """
    for attempt in range(MAX_RETRIES):
        try:
            response = client.models.embed_content(
                model=EMBEDDING_MODEL,
                contents=texts,
                config=types.EmbedContentConfig(output_dimensionality=OUTPUT_DIM),
            )
            return [list(e.values) for e in response.embeddings]
        except Exception as e:
            err_str = str(e).lower()
            if "429" in err_str or "quota" in err_str or "rate" in err_str or "resource" in err_str:
                wait = 10 * (2 ** attempt)  # 10s, 20s, 40s, 80s, 160s, 320s
                print(f"    [embeddings] Rate limit hit. Waiting {wait}s (attempt {attempt + 1}/{MAX_RETRIES})...")
                time.sleep(wait)
            else:
                wait = 5 * (2 ** attempt)
                print(f"    [embeddings] API error: {str(e)[:120]}. Retrying in {wait}s...")
                time.sleep(wait)

    raise RuntimeError(f"Embedding failed after {MAX_RETRIES} retries.")


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """
    Embed all chunks in batches of BATCH_SIZE.
    Attaches 'embedding' key (list[float]) to each chunk dict in-place.
    Returns the same list with embeddings populated.
    """
    client = _get_client()
    total = len(chunks)
    print(f"  [embeddings] Embedding {total} chunks in batches of {BATCH_SIZE} (Gemini {EMBEDDING_MODEL})...")

    for batch_start in range(0, total, BATCH_SIZE):
        batch = chunks[batch_start: batch_start + BATCH_SIZE]
        texts = [c["content"] for c in batch]
        vectors = _embed_batch(client, texts)

        for chunk, vector in zip(batch, vectors):
            chunk["embedding"] = vector

        print(f"    [embeddings] Batch {batch_start // BATCH_SIZE + 1}: "
              f"{batch_start + len(batch)}/{total} done")

    sample = chunks[0]["embedding"] if chunks else []
    print(f"  [embeddings] Embedding dim: {len(sample)} (expected {EMBEDDING_DIM})")
    return chunks


def embed_sentences(sentences: list[str]) -> list[list[float]]:
    """
    Embed individual sentences for semantic chunking.
    Returns a list of embedding vectors aligned with input sentences.
    """
    client = _get_client()
    total = len(sentences)
    print(f"  [embeddings] Embedding {total} sentences for semantic chunking (Gemini)...")

    all_vectors: list[list[float]] = []
    for batch_start in range(0, total, BATCH_SIZE):
        batch = sentences[batch_start: batch_start + BATCH_SIZE]
        vectors = _embed_batch(client, batch)
        all_vectors.extend(vectors)
        print(f"    [embeddings] Sentences: {batch_start + len(batch)}/{total} done")

    return all_vectors


def embed_query(query: str) -> list[float]:
    """
    Embed a single query string for semantic search.
    Returns a single embedding vector.
    """
    client = _get_client()
    vectors = _embed_batch(client, [query])
    return vectors[0]


def insert_chunks_to_db(chunks: list[dict], conn) -> int:
    """
    Insert a list of chunk dicts (with embeddings) into document_chunks table.
    Uses execute_batch for efficiency.
    Returns the number of rows inserted.
    """
    import psycopg2.extras

    records = [
        (
            c["chunk_id"],
            c["strategy"],
            c["source_page"],
            c["char_start"],
            c["char_end"],
            c["content"],
            c["token_count"],
            c["embedding"],
        )
        for c in chunks
    ]

    sql = """
        INSERT INTO document_chunks
            (chunk_id, strategy, source_page, char_start, char_end,
             content, token_count, embedding)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s::vector)
        ON CONFLICT (chunk_id) DO NOTHING;
    """

    cur = conn.cursor()
    psycopg2.extras.execute_batch(cur, sql, records, page_size=50)
    conn.commit()
    count = cur.rowcount
    cur.close()

    print(f"  [embeddings] Inserted {len(records)} chunks into document_chunks "
          f"({count} new rows, duplicates skipped)")
    return len(records)


def verify_row_counts(conn) -> dict[str, int]:
    """Print and return row counts per strategy."""
    cur = conn.cursor()
    cur.execute("""
        SELECT strategy, COUNT(*) as count
        FROM document_chunks
        GROUP BY strategy
        ORDER BY strategy;
    """)
    rows = cur.fetchall()
    cur.close()

    counts = {}
    print("\n--- Verification: row counts per strategy ---")
    for strategy, count in rows:
        print(f"  {strategy}: {count} rows")
        counts[strategy] = count
    return counts
