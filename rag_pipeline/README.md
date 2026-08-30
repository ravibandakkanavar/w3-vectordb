# W03 RAG Pipeline — Supabase + pgvector Chunking Comparison

A fully runnable Python pipeline that ingests a PDF, applies **3 chunking strategies**, generates embeddings, and produces a **retrieval quality comparison table** using Supabase with pgvector.

Built from the [W03 Meta-Prompting Playbook](../w03-meta-prompting-playbook.md) — this is the Stage 2 implementation of the [technical plan](../w03-rag-implementation-plan.md).

---

## Project Structure

```
rag_pipeline/
├── main.py           # Orchestrator — run this
├── db_setup.py       # Phase 1: Supabase schema + HNSW index
├── ingest.py         # Phase 2: PDF extraction + cleaning (pdfplumber)
├── chunkers.py       # Phase 3: chunk_fixed / chunk_structural / chunk_semantic
├── embeddings.py     # Phase 4: OpenAI text-embedding-3-small, batch insert
├── query.py          # Phase 5+6: semantic search, hit@3 scoring, comparison table
├── requirements.txt  # Pinned dependencies
├── .env.template     # Credentials template — copy to .env and fill in
├── data/             # Place your PDF here
└── results/          # Output: raw_results.json + comparison_table.md
```

---

## Quickstart

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Set up credentials

```bash
cp .env.template .env
# Edit .env and fill in SUPABASE_URL, SUPABASE_SERVICE_KEY, OPENAI_API_KEY, PDF_PATH
```

### 3. Place your PDF

```
rag_pipeline/data/document.pdf
```
Or set `PDF_PATH` in `.env` to any path.

### 4. Run the full pipeline

```bash
python main.py
```

Or run phases individually:

```bash
python main.py --phase setup    # Create DB table + HNSW index
python main.py --phase embed    # Ingest PDF → chunk → embed → insert to Supabase
python main.py --phase query    # Run 10 queries × 3 strategies → comparison table
```

---

## Before Running `--phase query`

Open `query.py` and replace the placeholder `QUERIES` and `GOLD_LABELS` with document-specific questions and pre-labeled gold chunk IDs. The pipeline will warn you if placeholders are still present, but results won't be meaningful until they're filled in.

---

## Chunking Strategies

| Strategy | Function | Parameters |
|---|---|---|
| Fixed-size with overlap | `chunk_fixed()` | 500 tokens, 50 overlap |
| Structural / recursive | `chunk_structural()` | 400 token target, 100 min |
| Semantic / sentence-window | `chunk_semantic()` | window=3, threshold=0.85 |

## Evaluation

- **Metric:** Hit-Rate@3 — did the pre-labeled gold chunk appear in the top-3 retrieved results?
- **Output:** `results/comparison_table.md` — 10-row × 3-column markdown table + summary
- **Raw data:** `results/raw_results.json` — full result dicts for Stage 3 analysis

## Database

- **Table:** `document_chunks` with `VECTOR(1536)` embedding column
- **Index:** HNSW (`m=16`, `ef_construction=64`) via pgvector
- **Why HNSW over IVFFlat:** a 50-page PDF produces ~150–600 chunks, well below IVFFlat's minimum row threshold for reliable recall

## Dependencies

| Package | Purpose |
|---|---|
| `pdfplumber` | Layout-aware PDF text extraction |
| `tiktoken` | Token counting (cl100k_base, matches OpenAI models) |
| `nltk` | Sentence tokenization for semantic chunking |
| `openai` | Embedding generation (text-embedding-3-small) |
| `psycopg2-binary` | Direct Postgres connection for pgvector queries |
| `supabase-py` | Supabase client (auth / REST) |
| `numpy` | Cosine similarity computation in semantic chunker |
| `python-dotenv` | `.env` credential loading |
