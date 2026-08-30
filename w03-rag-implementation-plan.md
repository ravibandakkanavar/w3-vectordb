# W03 RAG Pipeline — Technical Implementation Plan

> Generated from the W03 Meta-Prompting Playbook (Stage 1 output)

---

## Phase 1: Schema & Index

### Table Schema

```sql
CREATE TABLE document_chunks (
    id            BIGSERIAL PRIMARY KEY,
    chunk_id      TEXT NOT NULL UNIQUE,        -- e.g. "fixed_042", "struct_017", "sem_031"
    strategy      TEXT NOT NULL,               -- "fixed" | "structural" | "semantic"
    source_page   INTEGER NOT NULL,            -- 1-indexed page number from PDF
    char_start    INTEGER NOT NULL,            -- character offset in cleaned full text
    char_end      INTEGER NOT NULL,            -- character offset in cleaned full text
    content       TEXT NOT NULL,               -- raw chunk text
    token_count   INTEGER,                     -- for verification/debugging
    embedding     VECTOR(1536) NOT NULL        -- matches model dimension (see Phase 4)
);
```

> ASSUMPTION: `text-embedding-3-small` (OpenAI) will be used, which outputs 1536 dimensions. If you switch to `sentence-transformers/all-MiniLM-L6-v2`, change to `VECTOR(384)`.

### Index Decision: HNSW ✅ (not IVFFlat)

**Chosen:** HNSW

**Justification:**
- A 50-page PDF yields roughly 150–600 chunks total across all 3 strategies. IVFFlat requires a meaningful number of rows to train its inverted lists — the rule of thumb is at least `lists * 39` rows, and with default `lists=100` that means 3,900+ rows minimum for reliable recall. This dataset is well below that threshold.
- HNSW builds its graph incrementally with no minimum row count requirement and no training step.
- At this scale, HNSW's memory overhead (~few MB) is completely negligible.
- HNSW gives better recall at low row counts without tuning.

**Exact parameters:**

```sql
CREATE INDEX ON document_chunks
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);
```

- `m = 16`: number of bidirectional links per node. 16 is the standard default — sufficient for this dataset size, good recall/memory balance.
- `ef_construction = 64`: size of the dynamic candidate list during index build. Higher = better recall at index build time. 64 is the recommended minimum; no need to go higher for a few hundred vectors.
- `vector_cosine_ops`: cosine distance is the correct operator for normalized sentence embeddings.

---

## Phase 2: Ingestion

### PDF Extraction Library

**Chosen:** `pdfplumber`

**Justification:** Layout-aware extraction that preserves reading order and can detect headers/footers by bounding-box position. PyPDF2 is raw-text only and loses layout. pdfminer is lower-level. pdfplumber gives the right balance of control and usability without requiring OCR.

> ASSUMPTION: The PDF is text-based, not a scanned image.
> ASSUMPTION: The PDF has a consistent header/footer region (e.g. page numbers, document title) in the top/bottom ~10% of each page's bounding box.

### Extraction Approach

1. Extract text page-by-page using `pdfplumber.open()`, iterating `page.extract_text()`
2. Record page boundaries — track cumulative character offsets so `char_start/char_end` can map back to the full document string

### Cleaning Steps (in order)

| Step | What | Why |
|---|---|---|
| Header/footer removal | Drop lines matching a regex for page numbers or document title, OR use bounding-box crop to exclude top/bottom 8% of page height | Prevents chunk boundaries being polluted with navigation text |
| Hyphenation repair | Regex: `(\w+)-\n(\w+)` → `\1\2` | pdfplumber often preserves line-end hyphens as literal characters |
| Whitespace normalization | Collapse `\s+` to single space; strip leading/trailing whitespace per paragraph | Consistent tokenization |
| Ligature/encoding fix | Replace common PDF encoding artifacts (e.g. `ﬁ` → `fi`, `ﬀ` → `ff`) | Embedding models tokenize these incorrectly |
| Empty page filter | Skip pages where extracted text length < 20 chars | Avoids blank separator pages creating empty chunks |

---

## Phase 3: Three Chunking Strategies

### Strategy A — Fixed-Size with Overlap

**Function name:** `chunk_fixed(text, chunk_size=500, overlap=50)`

- Unit: **tokens** (using `tiktoken` with `cl100k_base` encoding, which matches `text-embedding-3-small`)
- Chunk size: **500 tokens**
- Overlap: **50 tokens** (10% — standard starting point, enough to preserve sentence context at boundaries)
- Procedure: tokenize the full cleaned text → slide a window of 500 tokens, step by 450 tokens, decode each window back to string
- `char_start/char_end`: map token offsets back to character offsets using tiktoken's offset tracking

> ASSUMPTION: 500-token chunks will produce ~100–200 chunks for a 50-page PDF (typical PDF page ≈ 250–350 tokens of body text).

### Strategy B — Structural / Recursive

**Function name:** `chunk_structural(text, target_size=400, min_size=100)`

- Split hierarchy (try in order, fall back to next if chunk still exceeds `target_size`):
  1. Split on `\n\n` (paragraph breaks)
  2. Split on `\n` (line breaks)
  3. Split on `. ` (sentence boundary)
  4. Hard split at `target_size` tokens (last resort fallback)
- Target size: **400 tokens**
- Merge rule: if a structural unit is < `min_size` (100 tokens), merge it with the next unit before checking size

> ASSUMPTION: The PDF has consistent paragraph breaks detectable as double newlines after pdfplumber extraction.

### Strategy C — Semantic / Sentence-Window

**Function name:** `chunk_semantic(sentences, window=3, similarity_threshold=0.85)`

- Approach: **sentence-window with embedding similarity**
- Split the cleaned text into individual sentences using `nltk.sent_tokenize`
- Embed each sentence individually (same model as Phase 4, batched)
- Build chunks by a sliding window of **3 sentences**, computing cosine similarity between adjacent windows
- When similarity between window N and window N+1 drops below **0.85**, treat it as a topic boundary and start a new chunk
- Merge windows within a topic into one chunk; if the resulting chunk exceeds 600 tokens, split at the nearest sentence boundary

> ASSUMPTION: 0.85 cosine similarity threshold is a reasonable starting point for `text-embedding-3-small`. Adjust to 0.80 or 0.90 if too many/few boundaries are detected. Flag the actual threshold used in the comparison table.

---

## Phase 4: Embeddings

### Model Choice

| Property | Value |
|---|---|
| Model | `text-embedding-3-small` (OpenAI API) |
| Dimension | 1536 |
| Fallback (offline) | `sentence-transformers/all-MiniLM-L6-v2` (384-dim) |

**Justification:** Strong benchmark performance, widely available, cost-effective at this scale (~600 chunks = negligible API cost). If using the fallback, change schema to `VECTOR(384)`.

> ASSUMPTION: An OpenAI API key is available as the `OPENAI_API_KEY` env var.

### Batching Approach

- Batch size: **100 chunks per API call**
- Use `openai.embeddings.create(input=[...], model="text-embedding-3-small")`
- On rate limit (429): exponential backoff — wait `2^n` seconds, max 3 retries

### Tagging Back to Metadata

Each embedding is inserted into `document_chunks` together with its metadata in a single `INSERT` — no separate tagging step. The `chunk_id` is constructed before embedding as `f"{strategy}_{index:03d}"`.

```
chunk_fixed_042  →  strategy="fixed",      source_page=7, char_start=14200, char_end=16800
chunk_struct_017 →  strategy="structural", source_page=3, ...
chunk_sem_031    →  strategy="semantic",   source_page=5, ...
```

---

## Phase 5: Query Design

### Design Principle

Queries must stress-test the structural differences between strategies. Fixed chunks may split a fact across a boundary; structural chunks preserve paragraph context; semantic chunks group by topic. Design queries so these differences surface in retrieval scores.

### 10 Query Templates

| # | Type | Query Pattern | What it differentiates |
|---|---|---|---|
| Q1 | Single-fact lookup | "What is the exact value of [specific statistic/number in the doc]?" | Fixed may split number from its context; structural/semantic likely keeps it whole |
| Q2 | Single-fact lookup | "Who is [named entity] and what is their role?" | Tests whether a short factual sentence lands cleanly in a chunk |
| Q3 | Single-fact lookup | "What does [defined term] mean?" | Definition may sit at a paragraph boundary — structural should handle cleanly |
| Q4 | Multi-sentence / context-dependent | "Explain the relationship between [concept A] and [concept B]" | Requires adjacent sentences — semantic chunking's window should capture this |
| Q5 | Multi-sentence / context-dependent | "What evidence supports [claim made mid-paragraph]?" | Context spans 3–5 sentences — tests overlap effectiveness in fixed strategy |
| Q6 | Multi-sentence / context-dependent | "What are the steps in [process described over a paragraph]?" | Sequential steps — structural chunking by paragraph should win here |
| Q7 | Cross-section synthesis | "How does [topic in section 2] connect to [topic in section 5]?" | All strategies will struggle; reveals score ceiling |
| Q8 | Cross-section synthesis | "What are all the limitations mentioned throughout the document?" | Dispersed content — tests whether any strategy aggregates across sections |
| Q9 | Cross-section synthesis | "Compare the approach in [early section] to [later section]" | Requires multi-hop retrieval — reveals top-k recall differences |
| Q10 | Adversarial / boundary | "What is discussed immediately after [mid-page concept]?" | Directly tests chunk boundary placement — fixed overlap vs structural split |

> **Before running queries:** manually identify the gold chunk for each question and record it. Required for hit-rate metric in Phase 6.

---

## Phase 6: Evaluation Method

### Primary Metric: Hit-Rate@3 Against Manually-Identified Gold Chunks

**Definition:** For a given query and strategy, did the correct chunk (pre-identified by human review) appear in the top-3 retrieved results? Score = 1 if yes, 0 if no.

**Why this metric:**
- Cosine similarity score alone is a weak proxy — a higher score doesn't mean the retrieved chunk is actually relevant
- Manual relevance ratings (1–5) are subjective and time-consuming for 30 query-strategy pairs
- Hit-rate@3 is binary, reproducible, and honest: gold chunks are identified *before* running queries, eliminating post-hoc rationalization

### Procedure

**Step 1 — Pre-labeling (before any queries run)**

For each of the 10 queries, read the PDF and record the `chunk_id` of the single chunk that best answers it:

```python
gold_labels = {
    "Q1": "fixed_042",
    "Q2": "fixed_017",
    # ...
}
```

> ASSUMPTION: Gold labels are defined on the fixed-strategy chunks first, then cross-referenced to structural/semantic by `char_start/char_end` overlap (>80% character overlap = same gold answer region).

**Step 2 — Query Execution (30 total = 10 queries × 3 strategies)**

```sql
SELECT chunk_id, content, 1 - (embedding <=> query_vector) AS score
FROM document_chunks
WHERE strategy = '<strategy>'
ORDER BY embedding <=> query_vector
LIMIT 3;
```

**Step 3 — Scoring**

Check if the gold chunk (or a chunk with >80% character overlap with it) appears in the top-3 results. Record 1 or 0.

**Secondary metric (reported alongside):** Top-1 cosine similarity score — report the raw score so the table shows both hit/miss and score.

### Output Format

Results stored as a list of dicts for Stage 3 consumption:

```python
{
  "query": "Q1",
  "strategy": "fixed",
  "retrieved_chunks": ["fixed_042", "fixed_039", "fixed_045"],
  "scores": [0.91, 0.88, 0.84],
  "hit_at_3": 1
}
```

### Markdown Comparison Table

```markdown
| Query | Type | Fixed (hit@3 / score) | Structural (hit@3 / score) | Semantic (hit@3 / score) |
|---|---|---|---|---|
| Q1: [short label] | Fact lookup      | ✅ 0.91 | ✅ 0.89 | ❌ 0.74 |
| Q2: [short label] | Fact lookup      | ...     | ...     | ...     |
| Q3: [short label] | Fact lookup      | ...     | ...     | ...     |
| Q4: [short label] | Multi-sentence   | ...     | ...     | ...     |
| Q5: [short label] | Multi-sentence   | ...     | ...     | ...     |
| Q6: [short label] | Multi-sentence   | ...     | ...     | ...     |
| Q7: [short label] | Cross-section    | ...     | ...     | ...     |
| Q8: [short label] | Cross-section    | ...     | ...     | ...     |
| Q9: [short label] | Cross-section    | ...     | ...     | ...     |
| Q10: [short label]| Boundary         | ...     | ...     | ...     |
| **Total hit@3**   |                  | **/10** | **/10** | **/10** |
```

Each cell = ✅/❌ + top-1 cosine score. Bottom row = total hit count across all 10 queries.

---

## Phase 7: Checklist — Success Criteria Mapping

| Success Criterion | Phase(s) | Verification Deliverable |
|---|---|---|
| ✅ Supabase pgvector configured with HNSW or IVFFlat index | Phase 1 | `document_chunks` table created with `VECTOR(1536)` column; HNSW index verified via `SELECT * FROM pg_indexes WHERE tablename='document_chunks'` |
| ✅ 3 chunking strategies implemented and compared | Phase 3 + 6 | `chunk_fixed()`, `chunk_structural()`, `chunk_semantic()` each produce rows in `document_chunks` with distinct `strategy` values; row counts per strategy logged |
| ✅ 10 semantic queries with documented results | Phase 5 + 6 | 10 queries × 3 strategies = 30 result sets, each with `retrieved_chunks` list and cosine scores, stored as structured output |
| ✅ Markdown comparison table of retrieval quality | Phase 6 | 10-row × 3-column table with hit@3 + cosine score per cell, total row, and summary paragraph |

---

## Environment & Dependencies

```
SUPABASE_URL=<your-supabase-project-url>
SUPABASE_SERVICE_KEY=<your-supabase-service-key>
OPENAI_API_KEY=<your-openai-api-key>
```

```
pdfplumber
tiktoken
nltk
openai>=1.0.0
supabase-py
psycopg2-binary
numpy
```

---

*This plan is the Stage 1 output of the W03 Meta-Prompting Playbook. Paste it into the Stage 2 prompt to begin code generation.*
