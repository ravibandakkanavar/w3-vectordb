"""
Phase 4: Embedding generation using OpenAI text-embedding-3-small.

Provides:
    embed_chunks()    — embed a list of chunk dicts, returns them with embeddings attached
    embed_sentences() — embed individual sentences (for semantic chunking)
    embed_query()     — embed a single query string for retrieval

Batching: 100 chunks per API call.
Retry: exponential backoff on rate limit (429), max 3 retries.
"""

import os
import time
import openai
from dotenv import load_dotenv

load_dotenv()

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM   = 1536
BATCH_SIZE      = 100
MAX_RETRIES     = 3


def _get_client() -> openai.OpenAI:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY is not set. Add it to your .env file.")
    return openai.OpenAI(api_key=api_key)


def _embed_batch(client: openai.OpenAI, texts: list[str]) -> list[list[float]]:
    """
    Embed a batch of texts with exponential backoff on rate limit errors.
    Returns a list of embedding vectors aligned with input texts.
    """
    for attempt in range(MAX_RETRIES):
        try:
            response = client.embeddings.create(
                input=texts,
                model=EMBEDDING_MODEL,
            )
            # Response data is ordered by index
            vectors = [item.embedding for item in sorted(response.data, key=lambda x: x.index)]
            return vectors
        except openai.RateLimitError:
            wait = 2 ** attempt
            print(f"    [embeddings] Rate limit hit. Waiting {wait}s (attempt {attempt + 1}/{MAX_RETRIES})...")
            time.sleep(wait)
        except openai.APIError as e:
            print(f"    [embeddings] API error: {e}. Retrying...")
            time.sleep(2 ** attempt)

    raise RuntimeError(f"Embedding failed after {MAX_RETRIES} retries.")


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """
    Embed all chunks in batches of BATCH_SIZE.
    Attaches 'embedding' key (list[float]) to each chunk dict in-place.
    Returns the same list with embeddings populated.
    """
    client = _get_client()
    total = len(chunks)
    print(f"  [embeddings] Embedding {total} chunks in batches of {BATCH_SIZE}...")

    for batch_start in range(0, total, BATCH_SIZE):
        batch = chunks[batch_start: batch_start + BATCH_SIZE]
        texts = [c["content"] for c in batch]
        vectors = _embed_batch(client, texts)

        for chunk, vector in zip(batch, vectors):
            chunk["embedding"] = vector

        print(f"    [embeddings] Batch {batch_start // BATCH_SIZE + 1}: "
              f"{batch_start + len(batch)}/{total} done")

    # Verify shape
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
    print(f"  [embeddings] Embedding {total} sentences for semantic chunking...")

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
    Uses a single executemany for efficiency.
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
    psycopg2.extras.execute_batch(cur, sql, records, page_size=100)
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
