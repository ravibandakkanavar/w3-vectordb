"""
Phase 1: Database setup — creates the document_chunks table and HNSW index.

Run this once before ingesting any data:
    python db_setup.py

Verification: queries pg_indexes to confirm the HNSW index was created.
"""

import os
import psycopg2
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Build a direct psycopg2 connection string from Supabase credentials.
# Supabase exposes a direct Postgres connection on port 5432.
# Connection string format: postgresql://postgres:<service_key>@db.<project-ref>.supabase.co:5432/postgres
# ---------------------------------------------------------------------------

def get_connection():
    supabase_url = os.environ["SUPABASE_URL"]          # e.g. https://abcdef.supabase.co
    db_password  = os.environ["SUPABASE_DB_PASSWORD"]  # direct Postgres password

    # Session pooler on port 5432 — supports DDL (CREATE EXTENSION, CREATE INDEX)
    project_ref = supabase_url.replace("https://", "").split(".")[0]

    conn = psycopg2.connect(
        host="aws-0-ap-northeast-2.pooler.supabase.com",
        port=5432,
        dbname="postgres",
        user=f"postgres.{project_ref}",
        password=db_password,
        sslmode="require",
        connect_timeout=30,
    )
    return conn


# ---------------------------------------------------------------------------
# SQL: enable pgvector, create table, create HNSW index
# ---------------------------------------------------------------------------

SQL_ENABLE_PGVECTOR = "CREATE EXTENSION IF NOT EXISTS vector;"

SQL_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS document_chunks (
    id            BIGSERIAL PRIMARY KEY,
    chunk_id      TEXT NOT NULL UNIQUE,
    strategy      TEXT NOT NULL CHECK (strategy IN ('fixed', 'structural', 'semantic')),
    source_page   INTEGER NOT NULL,
    char_start    INTEGER NOT NULL,
    char_end      INTEGER NOT NULL,
    content       TEXT NOT NULL,
    token_count   INTEGER,
    embedding     VECTOR(768) NOT NULL
);
"""

# HNSW index for 768-dim Gemini embeddings (gemini-embedding-001, output_dimensionality=768).
# vector_cosine_ops: cosine distance, correct for normalized embeddings.
SQL_CREATE_INDEX = """
CREATE INDEX IF NOT EXISTS document_chunks_embedding_hnsw_idx
ON document_chunks
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);
"""

SQL_VERIFY_INDEX = """
SELECT indexname, indexdef
FROM pg_indexes
WHERE tablename = 'document_chunks';
"""


def setup_database():
    print("Connecting to Supabase Postgres...")
    conn = get_connection()
    conn.autocommit = True
    cur = conn.cursor()

    print("Enabling pgvector extension...")
    cur.execute(SQL_ENABLE_PGVECTOR)

    print("Dropping existing table if present (schema change: 1536 → 3072 dims)...")
    cur.execute("DROP TABLE IF EXISTS document_chunks CASCADE;")

    print("Creating document_chunks table (VECTOR(3072) for Gemini embeddings)...")
    cur.execute(SQL_CREATE_TABLE)

    print("Creating HNSW index (m=16, ef_construction=64)...")
    cur.execute(SQL_CREATE_INDEX)

    # Verification
    cur.execute(SQL_VERIFY_INDEX)
    rows = cur.fetchall()
    print("\n--- Verification: pg_indexes for document_chunks ---")
    for indexname, indexdef in rows:
        print(f"  {indexname}: {indexdef}")

    cur.close()
    conn.close()
    print("\nDatabase setup complete.")


if __name__ == "__main__":
    setup_database()
