"""GET /api/status — chunk counts per strategy and DB health."""
from fastapi import APIRouter
from db import get_connection

router = APIRouter()


@router.get("/status")
def get_status():
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute("""
            SELECT strategy, COUNT(*) as count
            FROM document_chunks
            GROUP BY strategy
            ORDER BY strategy;
        """)
        rows = cur.fetchall()
        cur.close()
        conn.close()
        counts = {row[0]: row[1] for row in rows}
        total = sum(counts.values())
        return {"status": "ok", "total_chunks": total, "by_strategy": counts}
    except Exception as e:
        return {"status": "error", "detail": str(e)}
