"""
Main orchestrator — runs the full W03 RAG pipeline end-to-end.

Phases:
    1. DB setup     — create table + HNSW index (idempotent)
    2. Ingestion    — extract + clean PDF text
    3. Chunking     — apply 3 strategies
    4. Embeddings   — generate + insert into Supabase
    5. Queries      — run 10 semantic queries × 3 strategies
    6. Evaluation   — score hit@3, build + save comparison table

Usage:
    # Copy and fill in your credentials first:
    cp .env.template .env

    # Install dependencies:
    pip install -r requirements.txt

    # Place your PDF at the path set in .env (default: ./data/document.pdf)

    # Run the full pipeline:
    python main.py

    # Or run specific phases only:
    python main.py --phase setup
    python main.py --phase ingest
    python main.py --phase embed
    python main.py --phase query

IMPORTANT — before running --phase query:
    Open query.py and replace the QUERIES and GOLD_LABELS placeholders with
    document-specific questions and pre-labeled gold chunk IDs.
"""

import argparse
import os
import sys

from dotenv import load_dotenv

load_dotenv()


# ---------------------------------------------------------------------------
# DB connection helper (shared by embed and query phases)
# ---------------------------------------------------------------------------

def get_db_connection():
    import psycopg2
    supabase_url = os.environ["SUPABASE_URL"]
    service_key  = os.environ["SUPABASE_SERVICE_KEY"]
    project_ref  = supabase_url.replace("https://", "").split(".")[0]
    host         = f"db.{project_ref}.supabase.co"
    return psycopg2.connect(
        host=host, port=5432, dbname="postgres",
        user="postgres", password=service_key, sslmode="require",
    )


# ---------------------------------------------------------------------------
# Phase runners
# ---------------------------------------------------------------------------

def phase_setup():
    print("=" * 60)
    print("PHASE 1: Database setup")
    print("=" * 60)
    from db_setup import setup_database
    setup_database()


def phase_ingest() -> tuple[str, list[dict]]:
    print("\n" + "=" * 60)
    print("PHASE 2: PDF ingestion")
    print("=" * 60)
    from ingest import extract_full_text

    pdf_path = os.environ.get("PDF_PATH", "./data/document.pdf")
    if not os.path.exists(pdf_path):
        print(f"ERROR: PDF not found at '{pdf_path}'")
        print("Set PDF_PATH in your .env file or place the PDF at ./data/document.pdf")
        sys.exit(1)

    full_text, page_index = extract_full_text(pdf_path)
    print(f"  Full document: {len(full_text):,} characters across {len(page_index)} pages")
    return full_text, page_index


def phase_chunk(full_text: str, page_index: list[dict]) -> dict[str, list[dict]]:
    print("\n" + "=" * 60)
    print("PHASE 3: Chunking (3 strategies)")
    print("=" * 60)
    from chunkers import (
        chunk_fixed, chunk_structural, chunk_semantic, split_into_sentences
    )
    from embeddings import embed_sentences

    # Strategy A — fixed
    fixed_chunks = chunk_fixed(full_text, page_index)

    # Strategy B — structural
    structural_chunks = chunk_structural(full_text, page_index)

    # Strategy C — semantic (requires sentence embeddings first)
    print("  Splitting into sentences for semantic chunking...")
    sentences = split_into_sentences(full_text)
    print(f"  {len(sentences)} sentences found")

    print("  Embedding sentences (needed for topic-boundary detection)...")
    sentence_embeddings = embed_sentences(sentences)

    semantic_chunks = chunk_semantic(
        full_text, page_index, sentence_embeddings, sentences
    )

    all_chunks = {
        "fixed":      fixed_chunks,
        "structural": structural_chunks,
        "semantic":   semantic_chunks,
    }

    print("\n  Chunk count summary:")
    for strategy, chunks in all_chunks.items():
        token_counts = [c["token_count"] for c in chunks]
        avg = sum(token_counts) / len(token_counts) if token_counts else 0
        print(f"    {strategy}: {len(chunks)} chunks | avg {avg:.0f} tokens")

    return all_chunks


def phase_embed(all_chunks: dict[str, list[dict]]):
    print("\n" + "=" * 60)
    print("PHASE 4: Embeddings + DB insertion")
    print("=" * 60)
    from embeddings import embed_chunks, insert_chunks_to_db, verify_row_counts

    conn = get_db_connection()

    for strategy, chunks in all_chunks.items():
        print(f"\n  Embedding strategy: {strategy}")
        embedded = embed_chunks(chunks)
        insert_chunks_to_db(embedded, conn)

    verify_row_counts(conn)
    conn.close()


def phase_query():
    print("\n" + "=" * 60)
    print("PHASE 5 & 6: Queries + Evaluation")
    print("=" * 60)

    # Check placeholders haven't been left in
    from query import QUERIES, GOLD_LABELS
    placeholder_queries = [q for q in QUERIES.values() if q["text"].startswith("REPLACE:")]
    if placeholder_queries:
        print(
            f"\nWARNING: {len(placeholder_queries)} queries still have placeholder text.\n"
            "Open query.py and replace QUERIES and GOLD_LABELS with your document-specific values.\n"
            "Proceeding anyway — results will not be meaningful until placeholders are replaced.\n"
        )

    from query import (
        run_queries, score_results,
        build_comparison_table, save_results, save_comparison_table
    )

    conn = get_db_connection()

    results  = run_queries(conn)
    summary  = score_results(results)
    table_md = build_comparison_table(results)

    raw_path   = save_results(results)
    table_path = save_comparison_table(table_md)

    conn.close()

    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)
    print(f"  Raw results  → {raw_path}")
    print(f"  Comparison   → {table_path}")
    print("\n--- Final Scores ---")
    for strategy, data in summary.items():
        print(f"  {strategy}: {data['hit']}/{data['total']} hit@3 ({data['pct']}%)")

    print("\n--- Comparison Table Preview ---")
    print(table_md)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="W03 RAG Pipeline — Supabase + pgvector chunking comparison"
    )
    parser.add_argument(
        "--phase",
        choices=["all", "setup", "ingest", "embed", "query"],
        default="all",
        help=(
            "Which phase to run. "
            "'all' runs the full pipeline. "
            "'setup' creates the DB schema. "
            "'ingest' extracts + chunks the PDF. "
            "'embed' generates embeddings + inserts to DB. "
            "'query' runs queries + evaluation (edit QUERIES/GOLD_LABELS in query.py first)."
        ),
    )
    args = parser.parse_args()

    # Validate required env vars
    required_vars = ["SUPABASE_URL", "SUPABASE_SERVICE_KEY", "OPENAI_API_KEY"]
    missing = [v for v in required_vars if not os.environ.get(v)]
    if missing:
        print(f"ERROR: Missing required environment variables: {', '.join(missing)}")
        print("Copy .env.template to .env and fill in your credentials.")
        sys.exit(1)

    if args.phase in ("all", "setup"):
        phase_setup()

    if args.phase == "ingest":
        # Ingest-only: extract + display text, no embedding
        phase_ingest()
        print("\nIngestion complete. Run with --phase embed to generate embeddings.")

    elif args.phase in ("all", "embed"):
        full_text, page_index = phase_ingest()
        all_chunks = phase_chunk(full_text, page_index)
        phase_embed(all_chunks)

    if args.phase in ("all", "query"):
        phase_query()

    print("\nDone.")


if __name__ == "__main__":
    main()
