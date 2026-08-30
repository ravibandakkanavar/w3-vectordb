"""
Phase 5 & 6: Semantic search and hit-rate@3 evaluation.

Provides:
    run_queries()        — execute 10 queries × 3 strategies (30 searches total)
    score_results()      — compute hit@3 against pre-labeled gold chunks
    build_comparison_table() — produce the markdown comparison table
    save_results()       — persist raw results to results/raw_results.json

Query types (per the plan):
    Q1-Q3  : Single-fact lookup
    Q4-Q6  : Multi-sentence / context-dependent
    Q7-Q9  : Cross-section synthesis
    Q10    : Adversarial / boundary

IMPORTANT: Before running, fill in QUERIES below with document-specific questions
and fill in GOLD_LABELS with the chunk_ids that best answer each query.
Both are placeholders until you review your actual PDF.
"""

import json
import os
import psycopg2
import psycopg2.extras
from embeddings import embed_query


# ---------------------------------------------------------------------------
# EDIT THESE before running Phase 5
# Replace placeholder text with questions drawn from your actual PDF.
# ---------------------------------------------------------------------------

QUERIES: dict[str, dict] = {
    "Q1":  {"type": "Fact lookup",    "text": "REPLACE: What is the exact value of [key statistic in the document]?"},
    "Q2":  {"type": "Fact lookup",    "text": "REPLACE: Who is [named entity] and what is their role?"},
    "Q3":  {"type": "Fact lookup",    "text": "REPLACE: What does [defined term] mean according to the document?"},
    "Q4":  {"type": "Multi-sentence", "text": "REPLACE: Explain the relationship between [concept A] and [concept B]."},
    "Q5":  {"type": "Multi-sentence", "text": "REPLACE: What evidence supports [claim made mid-paragraph]?"},
    "Q6":  {"type": "Multi-sentence", "text": "REPLACE: What are the steps in [process described over a paragraph]?"},
    "Q7":  {"type": "Cross-section",  "text": "REPLACE: How does [topic in early section] connect to [topic in later section]?"},
    "Q8":  {"type": "Cross-section",  "text": "REPLACE: What are all the limitations mentioned throughout the document?"},
    "Q9":  {"type": "Cross-section",  "text": "REPLACE: Compare the approach described in [early section] to [later section]."},
    "Q10": {"type": "Boundary",       "text": "REPLACE: What is discussed immediately after [mid-page concept]?"},
}

# Pre-label gold chunks BEFORE running queries.
# For each query, identify the single chunk_id that best answers it by reading the PDF.
# Cross-reference structural/semantic by char_start/char_end overlap (>80% overlap = same region).
GOLD_LABELS: dict[str, str] = {
    "Q1":  "fixed_000",   # REPLACE with real chunk_id
    "Q2":  "fixed_000",
    "Q3":  "fixed_000",
    "Q4":  "fixed_000",
    "Q5":  "fixed_000",
    "Q6":  "fixed_000",
    "Q7":  "fixed_000",
    "Q8":  "fixed_000",
    "Q9":  "fixed_000",
    "Q10": "fixed_000",
}

STRATEGIES = ["fixed", "structural", "semantic"]
TOP_K = 3


# ---------------------------------------------------------------------------
# Core search
# ---------------------------------------------------------------------------

def search(
    conn,
    query_vector: list[float],
    strategy: str,
    top_k: int = TOP_K,
) -> list[dict]:
    """
    Run a cosine similarity search against document_chunks for a given strategy.
    Returns top_k results as dicts: {chunk_id, content, score, char_start, char_end}.
    """
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cur.execute(
        """
        SELECT
            chunk_id,
            content,
            source_page,
            char_start,
            char_end,
            1 - (embedding <=> %s::vector) AS score
        FROM document_chunks
        WHERE strategy = %s
        ORDER BY embedding <=> %s::vector
        LIMIT %s;
        """,
        (query_vector, strategy, query_vector, top_k),
    )
    rows = cur.fetchall()
    cur.close()
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Gold chunk matching (by chunk_id or char overlap)
# ---------------------------------------------------------------------------

def _overlaps_gold(result_chunk: dict, gold_chunk_id: str, all_chunks_index: dict) -> bool:
    """
    Returns True if result_chunk matches the gold chunk either:
    - Exactly by chunk_id, OR
    - By >80% character overlap with the gold chunk's char_start/char_end range
      (handles cross-strategy gold label matching)
    """
    if result_chunk["chunk_id"] == gold_chunk_id:
        return True

    gold = all_chunks_index.get(gold_chunk_id)
    if gold is None:
        return False

    # Compute character overlap ratio
    r_start = result_chunk["char_start"]
    r_end   = result_chunk["char_end"]
    g_start = gold["char_start"]
    g_end   = gold["char_end"]

    overlap_start = max(r_start, g_start)
    overlap_end   = min(r_end,   g_end)
    if overlap_end <= overlap_start:
        return False

    overlap_len = overlap_end - overlap_start
    gold_len    = g_end - g_start
    return (overlap_len / gold_len) >= 0.80 if gold_len > 0 else False


def _build_chunks_index(conn) -> dict[str, dict]:
    """Load all chunk metadata (no embeddings) into a dict keyed by chunk_id."""
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cur.execute("SELECT chunk_id, char_start, char_end FROM document_chunks;")
    rows = cur.fetchall()
    cur.close()
    return {r["chunk_id"]: dict(r) for r in rows}


# ---------------------------------------------------------------------------
# Run all queries
# ---------------------------------------------------------------------------

def run_queries(conn) -> list[dict]:
    """
    Execute all 10 queries × 3 strategies = 30 searches.
    Returns raw results list suitable for scoring and Stage 3 consumption.
    """
    chunks_index = _build_chunks_index(conn)
    results = []

    print(f"\n[query] Running {len(QUERIES)} queries × {len(STRATEGIES)} strategies...")

    for q_id, q_data in QUERIES.items():
        q_text = q_data["text"]
        q_type = q_data["type"]
        print(f"\n  {q_id} ({q_type}): {q_text[:80]}...")

        query_vector = embed_query(q_text)

        for strategy in STRATEGIES:
            hits = search(conn, query_vector, strategy, top_k=TOP_K)

            gold_id   = GOLD_LABELS.get(q_id, "")
            hit_at_3  = int(any(_overlaps_gold(h, gold_id, chunks_index) for h in hits))
            top_score = round(hits[0]["score"], 4) if hits else 0.0

            result = {
                "query_id":         q_id,
                "query_type":       q_type,
                "query_text":       q_text,
                "strategy":         strategy,
                "gold_chunk_id":    gold_id,
                "retrieved_chunks": [h["chunk_id"] for h in hits],
                "scores":           [round(h["score"], 4) for h in hits],
                "top_score":        top_score,
                "hit_at_3":         hit_at_3,
            }
            results.append(result)
            hit_icon = "✅" if hit_at_3 else "❌"
            print(f"    [{strategy}] {hit_icon} top score={top_score:.4f} | top chunk={hits[0]['chunk_id'] if hits else 'n/a'}")

    return results


# ---------------------------------------------------------------------------
# Scoring summary
# ---------------------------------------------------------------------------

def score_results(results: list[dict]) -> dict:
    """
    Compute hit@3 totals per strategy.
    Returns a dict: {strategy: {"hit": int, "total": int, "pct": float}}
    """
    summary = {s: {"hit": 0, "total": 0} for s in STRATEGIES}
    for r in results:
        s = r["strategy"]
        summary[s]["total"] += 1
        summary[s]["hit"]   += r["hit_at_3"]

    for s, data in summary.items():
        data["pct"] = round(data["hit"] / data["total"] * 100, 1) if data["total"] > 0 else 0.0

    print("\n--- Hit-Rate@3 Summary ---")
    for s, data in summary.items():
        print(f"  {s}: {data['hit']}/{data['total']} ({data['pct']}%)")

    return summary


# ---------------------------------------------------------------------------
# Markdown comparison table
# ---------------------------------------------------------------------------

def build_comparison_table(results: list[dict]) -> str:
    """
    Build the markdown comparison table:
    rows = 10 queries, columns = 3 strategies.
    Each cell = ✅/❌ + top-1 cosine score.
    Bottom row = total hit@3 per strategy.
    """
    # Index results by (query_id, strategy)
    index: dict[tuple, dict] = {}
    for r in results:
        index[(r["query_id"], r["strategy"])] = r

    lines = [
        "| Query | Type | Fixed (hit@3 / score) | Structural (hit@3 / score) | Semantic (hit@3 / score) |",
        "|---|---|---|---|---|",
    ]

    totals = {s: 0 for s in STRATEGIES}

    for q_id, q_data in QUERIES.items():
        q_type  = q_data["type"]
        q_label = q_id  # short label; replace with a meaningful phrase after reviewing results

        cells = []
        for strategy in STRATEGIES:
            r = index.get((q_id, strategy))
            if r:
                icon  = "✅" if r["hit_at_3"] else "❌"
                score = r["top_score"]
                cells.append(f"{icon} {score:.2f}")
                totals[strategy] += r["hit_at_3"]
            else:
                cells.append("—")

        lines.append(f"| {q_label} | {q_type} | {cells[0]} | {cells[1]} | {cells[2]} |")

    # Totals row
    total_cells = [f"**{totals[s]}/10**" for s in STRATEGIES]
    lines.append(f"| **Total hit@3** | | {total_cells[0]} | {total_cells[1]} | {total_cells[2]} |")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Persist results
# ---------------------------------------------------------------------------

def save_results(results: list[dict], output_dir: str = "./results") -> str:
    """Save raw results to JSON for Stage 3 consumption."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "raw_results.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n[query] Raw results saved to '{path}'")
    return path


def save_comparison_table(table_md: str, output_dir: str = "./results") -> str:
    """Save the markdown comparison table to a file."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "comparison_table.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("# RAG Retrieval Quality Comparison\n\n")
        f.write(f"**Metric:** Hit-Rate@3 (did the gold chunk appear in top-3 results?)\n\n")
        f.write(f"**Embedding model:** text-embedding-3-small (1536-dim)\n\n")
        f.write(table_md)
        f.write("\n\n## Limitations\n\n")
        f.write(
            "- Single document (50 pages): results may not generalise to larger corpora.\n"
            "- Single embedding model: a different model may favour different chunking strategies.\n"
            "- Gold labels were manually assigned by a single reviewer — no independently verified ground truth.\n"
            "- Sample size of 10 queries is too small for statistical significance.\n"
        )
    print(f"[query] Comparison table saved to '{path}'")
    return path
