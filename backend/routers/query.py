"""POST /api/query — semantic search across all 3 strategies."""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import psycopg2.extras

from db import get_connection
from embeddings import embed_single

router = APIRouter()

STRATEGIES = ["fixed", "structural", "semantic"]
TOP_K = 3


class QueryRequest(BaseModel):
    query: str


class ChunkResult(BaseModel):
    chunk_id: str
    content: str
    source_page: int
    score: float


class StrategyResult(BaseModel):
    strategy: str
    results: list[ChunkResult]


@router.post("/query")
def run_query(req: QueryRequest) -> list[StrategyResult]:
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query text cannot be empty.")

    # Embed the query
    try:
        query_vector = embed_single(req.query)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding failed: {e}")

    # Search each strategy
    try:
        conn = get_connection()
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

        strategy_results = []
        for strategy in STRATEGIES:
            cur.execute(
                """
                SELECT
                    chunk_id,
                    content,
                    source_page,
                    ROUND(CAST(1 - (embedding <=> %s::vector) AS numeric), 4) AS score
                FROM document_chunks
                WHERE strategy = %s
                ORDER BY embedding <=> %s::vector
                LIMIT %s;
                """,
                (query_vector, strategy, query_vector, TOP_K),
            )
            rows = cur.fetchall()
            strategy_results.append(StrategyResult(
                strategy=strategy,
                results=[ChunkResult(**dict(r)) for r in rows],
            ))

        cur.close()
        conn.close()
        return strategy_results

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {e}")
