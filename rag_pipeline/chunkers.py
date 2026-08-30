"""
Phase 3: Three chunking strategies.

Each function takes the cleaned full-document text (and page_index for metadata)
and returns a list of chunk dicts:
    {
        "chunk_id"   : str,   # e.g. "fixed_042"
        "strategy"   : str,   # "fixed" | "structural" | "semantic"
        "source_page": int,   # 1-indexed page number
        "char_start" : int,   # character offset in full doc string
        "char_end"   : int,   # character offset in full doc string
        "content"    : str,   # raw chunk text
        "token_count": int,   # number of tokens (cl100k_base)
    }

Strategy A — chunk_fixed   : fixed-size sliding window (500 tokens, 50 overlap)
Strategy B — chunk_structural: recursive paragraph/sentence split (target 400 tokens)
Strategy C — chunk_semantic  : sentence-window + embedding similarity (window=3, threshold=0.85)
"""

import re
import numpy as np
import tiktoken
import nltk
from ingest import char_offset_to_page

# Ensure NLTK sentence tokenizer data is available
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)

try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab", quiet=True)

# Tokenizer shared across all strategies (matches text-embedding-3-small)
TOKENIZER = tiktoken.get_encoding("cl100k_base")


def _tokenize(text: str) -> list[int]:
    return TOKENIZER.encode(text)


def _detokenize(tokens: list[int]) -> str:
    return TOKENIZER.decode(tokens)


def _token_count(text: str) -> int:
    return len(_tokenize(text))


# ---------------------------------------------------------------------------
# Strategy A — Fixed-size with overlap
# ---------------------------------------------------------------------------

def chunk_fixed(
    full_text: str,
    page_index: list[dict],
    chunk_size: int = 500,
    overlap: int = 50,
) -> list[dict]:
    """
    Slide a window of `chunk_size` tokens with `overlap` tokens of context carry-over.
    Step size = chunk_size - overlap = 450 tokens.

    char_start/char_end are approximated by finding the decoded chunk text in the
    full document string (first occurrence from expected position).
    """
    tokens = _tokenize(full_text)
    step = chunk_size - overlap
    chunks = []
    idx = 0
    chunk_num = 0
    search_start = 0  # scan position in full_text for char offset recovery

    while idx < len(tokens):
        window = tokens[idx: idx + chunk_size]
        text = _detokenize(window)

        # Recover char_start by searching forward from last known position
        char_start = full_text.find(text[:40], search_start)
        if char_start == -1:
            char_start = search_start  # fallback
        char_end = char_start + len(text)
        search_start = max(search_start, char_start + step * 3)  # advance hint

        source_page = char_offset_to_page(char_start, page_index)

        chunks.append({
            "chunk_id":    f"fixed_{chunk_num:03d}",
            "strategy":    "fixed",
            "source_page": source_page,
            "char_start":  char_start,
            "char_end":    char_end,
            "content":     text,
            "token_count": len(window),
        })
        chunk_num += 1
        idx += step

    print(f"  [chunkers] fixed: {len(chunks)} chunks (size={chunk_size}, overlap={overlap})")
    return chunks


# ---------------------------------------------------------------------------
# Strategy B — Structural / Recursive
# ---------------------------------------------------------------------------

def chunk_structural(
    full_text: str,
    page_index: list[dict],
    target_size: int = 400,
    min_size: int = 100,
) -> list[dict]:
    """
    Split hierarchy:
      1. Split on double newline (paragraph)
      2. Split on single newline (line)
      3. Split on '. ' (sentence)
      4. Hard token split at target_size (fallback)

    Merge units smaller than min_size into the next unit before size-checking.
    """

    def _hard_split(text: str, max_tokens: int) -> list[str]:
        """Split a text that exceeds max_tokens by hard token boundary."""
        tokens = _tokenize(text)
        parts = []
        for i in range(0, len(tokens), max_tokens):
            parts.append(_detokenize(tokens[i: i + max_tokens]))
        return parts

    def _recursive_split(text: str) -> list[str]:
        """Recursively split text until all pieces are <= target_size tokens."""
        if _token_count(text) <= target_size:
            return [text]

        # Try each separator in order
        for sep in ["\n\n", "\n", ". "]:
            parts = text.split(sep)
            if len(parts) > 1:
                # Rejoin separator into each part (except last)
                rejoined = []
                for i, p in enumerate(parts):
                    rejoined.append(p + (sep if i < len(parts) - 1 else ""))
                # Recursively split any part still too large
                result = []
                for part in rejoined:
                    result.extend(_recursive_split(part.strip()))
                return [r for r in result if r.strip()]

        # Fallback: hard token split
        return _hard_split(text, target_size)

    raw_units = _recursive_split(full_text)

    # Merge units smaller than min_size with the next unit
    merged: list[str] = []
    buffer = ""
    for unit in raw_units:
        if buffer:
            combined = buffer + " " + unit
            if _token_count(combined) <= target_size:
                buffer = combined
            else:
                merged.append(buffer.strip())
                buffer = unit
        else:
            if _token_count(unit) < min_size:
                buffer = unit
            else:
                merged.append(unit.strip())
    if buffer:
        merged.append(buffer.strip())

    # Build chunk dicts
    chunks = []
    search_start = 0
    for i, text in enumerate(merged):
        if not text.strip():
            continue
        char_start = full_text.find(text[:40], search_start)
        if char_start == -1:
            char_start = search_start
        char_end = char_start + len(text)
        search_start = max(search_start, char_start + 1)

        source_page = char_offset_to_page(char_start, page_index)
        chunks.append({
            "chunk_id":    f"structural_{i:03d}",
            "strategy":    "structural",
            "source_page": source_page,
            "char_start":  char_start,
            "char_end":    char_end,
            "content":     text,
            "token_count": _token_count(text),
        })

    print(f"  [chunkers] structural: {len(chunks)} chunks (target={target_size}, min={min_size})")
    return chunks


# ---------------------------------------------------------------------------
# Strategy C — Semantic / Sentence-window
# ---------------------------------------------------------------------------

def chunk_semantic(
    full_text: str,
    page_index: list[dict],
    sentence_embeddings: list[list[float]],  # pre-computed per-sentence embeddings
    sentences: list[str],                    # sentences aligned with embeddings
    window: int = 3,
    similarity_threshold: float = 0.85,
    max_chunk_tokens: int = 600,
) -> list[dict]:
    """
    Build chunks using a sliding window of `window` sentences.
    A topic boundary is detected when cosine similarity between adjacent windows
    drops below `similarity_threshold`. Chunks exceeding max_chunk_tokens are
    hard-split at the nearest sentence boundary.

    `sentence_embeddings` and `sentences` must be aligned (same index = same sentence).
    Call embeddings.embed_sentences() first, then pass results here.
    """

    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        a, b = np.array(a), np.array(b)
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        return float(np.dot(a, b) / denom) if denom > 0 else 0.0

    def _window_embedding(embs: list[list[float]]) -> list[float]:
        """Average embeddings within a window."""
        arr = np.array(embs)
        mean = arr.mean(axis=0)
        norm = np.linalg.norm(mean)
        return (mean / norm).tolist() if norm > 0 else mean.tolist()

    n = len(sentences)
    if n == 0:
        return []

    # Build window embeddings for each starting position
    window_embs = []
    for i in range(n):
        end = min(i + window, n)
        window_embs.append(_window_embedding(sentence_embeddings[i:end]))

    # Detect topic boundaries: where similarity between window i and window i+1
    # drops below threshold
    boundaries = [0]  # always start a new chunk at sentence 0
    for i in range(len(window_embs) - 1):
        sim = _cosine_similarity(window_embs[i], window_embs[i + 1])
        if sim < similarity_threshold:
            boundaries.append(i + 1)
    boundaries.append(n)  # sentinel

    # Build raw topic groups
    groups: list[list[str]] = []
    for b in range(len(boundaries) - 1):
        start, end = boundaries[b], boundaries[b + 1]
        groups.append(sentences[start:end])

    # Hard-split groups that exceed max_chunk_tokens
    final_groups: list[list[str]] = []
    for group in groups:
        combined = " ".join(group)
        if _token_count(combined) <= max_chunk_tokens:
            final_groups.append(group)
        else:
            # Split at sentence boundaries
            current: list[str] = []
            for sent in group:
                test = " ".join(current + [sent])
                if _token_count(test) > max_chunk_tokens and current:
                    final_groups.append(current)
                    current = [sent]
                else:
                    current.append(sent)
            if current:
                final_groups.append(current)

    # Build chunk dicts
    chunks = []
    search_start = 0
    for i, group in enumerate(final_groups):
        text = " ".join(group).strip()
        if not text:
            continue
        char_start = full_text.find(text[:40], search_start)
        if char_start == -1:
            char_start = search_start
        char_end = char_start + len(text)
        search_start = max(search_start, char_start + 1)

        source_page = char_offset_to_page(char_start, page_index)
        chunks.append({
            "chunk_id":    f"semantic_{i:03d}",
            "strategy":    "semantic",
            "source_page": source_page,
            "char_start":  char_start,
            "char_end":    char_end,
            "content":     text,
            "token_count": _token_count(text),
        })

    print(
        f"  [chunkers] semantic: {len(chunks)} chunks "
        f"(window={window}, threshold={similarity_threshold})"
    )
    return chunks


def split_into_sentences(text: str) -> list[str]:
    """Split cleaned full text into individual sentences using NLTK."""
    sentences = nltk.sent_tokenize(text)
    return [s.strip() for s in sentences if s.strip()]
