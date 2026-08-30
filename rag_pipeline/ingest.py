"""
Phase 2: PDF extraction and cleaning using pdfplumber.

Returns a list of page dicts:
    [
        {
            "page": 1,                  # 1-indexed
            "text": "cleaned text...",  # cleaned body text
            "char_offset": 0,           # cumulative char offset in full-doc string
        },
        ...
    ]

Also exposes extract_full_text() which returns the entire cleaned document as a
single string with page-boundary offsets tracked.
"""

import re
import pdfplumber


# ---------------------------------------------------------------------------
# Ligature / encoding artifact map (common PDF encoding issues)
# ---------------------------------------------------------------------------
LIGATURE_MAP = {
    "\ufb01": "fi",   # ﬁ
    "\ufb02": "fl",   # ﬂ
    "\ufb00": "ff",   # ﬀ
    "\ufb03": "ffi",  # ﬃ
    "\ufb04": "ffl",  # ﬄ
    "\u2013": "-",    # en dash
    "\u2014": "-",    # em dash
    "\u2018": "'",    # left single quote
    "\u2019": "'",    # right single quote
    "\u201c": '"',    # left double quote
    "\u201d": '"',    # right double quote
}


def _fix_ligatures(text: str) -> str:
    """Replace PDF ligature/encoding artifacts with ASCII equivalents."""
    for char, replacement in LIGATURE_MAP.items():
        text = text.replace(char, replacement)
    return text


def _repair_hyphenation(text: str) -> str:
    """Merge words split across lines with a hyphen (e.g. 'imple-\nmentation' → 'implementation')."""
    return re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)


def _remove_headers_footers(lines: list[str]) -> list[str]:
    """
    Heuristic header/footer removal:
    - Drop lines that are purely a page number (digits only, optionally with whitespace)
    - Drop lines shorter than 4 chars that appear at the very top or bottom of a page
    - Drop lines matching common header/footer patterns (e.g. "Chapter X", "Page X of Y")
    """
    if not lines:
        return lines

    def is_noise(line: str) -> bool:
        stripped = line.strip()
        if not stripped:
            return True
        # Pure page number
        if re.fullmatch(r"\d+", stripped):
            return True
        # "Page N" or "Page N of M"
        if re.fullmatch(r"[Pp]age\s+\d+(\s+of\s+\d+)?", stripped):
            return True
        # Very short line (likely a stray header artifact)
        if len(stripped) <= 3:
            return True
        return False

    # Remove first and last lines if they look like noise
    cleaned = list(lines)
    if cleaned and is_noise(cleaned[0]):
        cleaned = cleaned[1:]
    if cleaned and is_noise(cleaned[-1]):
        cleaned = cleaned[:-1]

    return cleaned


def _clean_page_text(raw_text: str) -> str:
    """
    Apply all cleaning steps to a single page's raw text:
    1. Fix ligatures/encoding artifacts
    2. Repair line-end hyphenation
    3. Remove header/footer lines
    4. Normalize whitespace
    """
    if not raw_text:
        return ""

    text = _fix_ligatures(raw_text)
    text = _repair_hyphenation(text)

    lines = text.split("\n")
    lines = _remove_headers_footers(lines)
    text = "\n".join(lines)

    # Normalize whitespace: collapse runs of spaces/tabs (but keep newlines)
    text = re.sub(r"[ \t]+", " ", text)
    # Collapse 3+ consecutive newlines to 2 (preserve paragraph breaks)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = text.strip()

    return text


def extract_pages(pdf_path: str) -> list[dict]:
    """
    Extract and clean text from each page of a PDF.

    Returns a list of page dicts with keys:
        page        : 1-indexed page number
        text        : cleaned text for this page
        char_offset : cumulative character offset (start of this page in full doc)
    """
    pages = []
    cumulative_offset = 0

    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            raw_text = page.extract_text() or ""

            # Skip effectively empty pages
            if len(raw_text.strip()) < 20:
                print(f"  [ingest] Page {page_num}: skipped (empty/short)")
                continue

            cleaned = _clean_page_text(raw_text)

            pages.append({
                "page": page_num,
                "text": cleaned,
                "char_offset": cumulative_offset,
            })

            # +1 for the newline separator we add when joining pages
            cumulative_offset += len(cleaned) + 1

    print(f"  [ingest] Extracted {len(pages)} non-empty pages from '{pdf_path}'")
    return pages


def extract_full_text(pdf_path: str) -> tuple[str, list[dict]]:
    """
    Returns:
        full_text   : entire cleaned document as a single string (pages joined with '\n')
        page_index  : list of page dicts (from extract_pages) for offset lookups
    """
    page_index = extract_pages(pdf_path)
    full_text = "\n".join(p["text"] for p in page_index)
    return full_text, page_index


def char_offset_to_page(char_pos: int, page_index: list[dict]) -> int:
    """
    Given a character position in the full document string, return the 1-indexed
    source page number. Falls back to the last page if beyond all offsets.
    """
    source_page = page_index[-1]["page"] if page_index else 1
    for i, page in enumerate(page_index):
        next_offset = page_index[i + 1]["char_offset"] if i + 1 < len(page_index) else float("inf")
        if page["char_offset"] <= char_pos < next_offset:
            source_page = page["page"]
            break
    return source_page


if __name__ == "__main__":
    import sys
    import os
    from dotenv import load_dotenv

    load_dotenv()
    pdf_path = os.environ.get("PDF_PATH", "./data/document.pdf")

    if not os.path.exists(pdf_path):
        print(f"PDF not found at '{pdf_path}'. Place your PDF there and retry.")
        sys.exit(1)

    full_text, page_index = extract_full_text(pdf_path)
    print(f"\nFull document length: {len(full_text)} characters")
    print(f"Pages extracted: {len(page_index)}")
    print("\nFirst 500 chars of cleaned text:\n")
    print(full_text[:500])
