"""POST /api/ingest — upload a PDF, chunk it 3 ways, embed and store."""
import sys, os, tempfile

# Add rag_pipeline to path for ingest/chunkers modules
RAG_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "rag_pipeline")
BACKEND_PATH = os.path.join(os.path.dirname(__file__), "..")
# Backend path must come first so backend/embeddings.py takes priority
if BACKEND_PATH not in sys.path:
    sys.path.insert(0, BACKEND_PATH)
if RAG_PATH not in sys.path:
    sys.path.append(RAG_PATH)

from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
import psycopg2.extras

from db import get_connection
from embeddings import embed_texts

router = APIRouter()

# Simple in-memory progress tracker
_progress: dict = {"status": "idle", "message": "", "detail": {}}


def _get_progress():
    return _progress.copy()


def _set_progress(status: str, message: str, detail: dict = {}):
    _progress["status"] = status
    _progress["message"] = message
    _progress["detail"] = detail


@router.get("/ingest/progress")
def ingest_progress():
    return _get_progress()


def _run_ingest(pdf_bytes: bytes):
    import tempfile, os
    from ingest import extract_full_text
    from chunkers import chunk_fixed, chunk_structural, chunk_semantic, split_into_sentences

    try:
        _set_progress("running", "Extracting PDF text...")

        # Write PDF to a temp file
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(pdf_bytes)
            tmp_path = tmp.name

        full_text, page_index = extract_full_text(tmp_path)
        os.unlink(tmp_path)

        _set_progress("running", f"Extracted {len(full_text):,} characters. Chunking...")

        # Strategy A — fixed
        fixed_chunks = chunk_fixed(full_text, page_index)

        # Strategy B — structural
        structural_chunks = chunk_structural(full_text, page_index)

        # Strategy C — semantic (needs sentence embeddings)
        _set_progress("running", "Embedding sentences for semantic chunking...")
        sentences = split_into_sentences(full_text)
        sentence_vectors = embed_texts(sentences)
        semantic_chunks = chunk_semantic(full_text, page_index, sentence_vectors, sentences)

        all_chunks = fixed_chunks + structural_chunks + semantic_chunks
        _set_progress("running", f"Chunked into {len(all_chunks)} total chunks. Embedding...")

        # Embed all chunks
        texts = [c["content"] for c in all_chunks]
        vectors = embed_texts(texts)
        for chunk, vector in zip(all_chunks, vectors):
            chunk["embedding"] = vector

        # Insert into DB
        _set_progress("running", "Inserting into Supabase...")
        conn = get_connection()

        # Clear existing data first
        cur = conn.cursor()
        cur.execute("DELETE FROM document_chunks;")

        records = [
            (c["chunk_id"], c["strategy"], c["source_page"],
             c["char_start"], c["char_end"], c["content"],
             c["token_count"], c["embedding"])
            for c in all_chunks
        ]
        sql = """
            INSERT INTO document_chunks
                (chunk_id, strategy, source_page, char_start, char_end,
                 content, token_count, embedding)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s::vector)
            ON CONFLICT (chunk_id) DO NOTHING;
        """
        psycopg2.extras.execute_batch(cur, sql, records, page_size=50)
        conn.commit()
        cur.close()
        conn.close()

        counts = {
            "fixed": len(fixed_chunks),
            "structural": len(structural_chunks),
            "semantic": len(semantic_chunks),
        }
        _set_progress("done", "Ingestion complete.", counts)

    except Exception as e:
        _set_progress("error", str(e))


@router.post("/ingest")
async def ingest_pdf(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    if _progress["status"] == "running":
        raise HTTPException(status_code=409, detail="Ingestion already in progress.")

    pdf_bytes = await file.read()
    _set_progress("running", "Starting ingestion...")
    background_tasks.add_task(_run_ingest, pdf_bytes)
    return {"message": "Ingestion started.", "filename": file.filename}
