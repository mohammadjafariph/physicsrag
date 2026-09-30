"""
Stage 6 — physics-aware document chunker.

Splits a LoadedDocument (Stage 5) into section-aware Chunks:

    LoadedDocument
        -> [Chunk {chunk_id, paper_id, title, section, text, page, index}]

Heading detection is TYPOGRAPHIC first, textual second:
1. Font-based: a line set in a font noticeably larger than the body text,
   or bold and short, is a heading (works across all LaTeX styles).
2. Regex fallback: "I. Introduction", "2.1 Subsection", "A. Details",
   "ALL CAPS" patterns catch headings that share the body font.

Page 1 is exempt from font detection (title/author lines are also large
or bold); regexes still apply there.
"""

import re
from collections import Counter
from dataclasses import dataclass

from src.documents.loader import Line, LoadedDocument

TARGET_CHUNK_CHARS = 1200
MIN_CHUNK_CHARS = 150  # shorter chunks are merged into the previous one

BODY_SIZE_FACTOR = 1.12   # heading font must exceed body font by this ratio
MAX_HEADING_WORDS = 15    # headings are short; long "large" lines are titles

# Lines starting with these are captions/labels, never headings.
_CAPTION_PREFIXES = ("fig.", "figure", "table", "tab.")

# A numbered heading must not end mid-sentence ("2. ... equivalent to").
_ENDS_MID_SENTENCE = re.compile(
    r"\b(?:to|of|in|and|the|a|an|with|for|by|on|is|are|that|which)$"
)

_MATH_CHARS = set("=()[]{}|∇∂ρψ√×·≤≥→~")

# Fallback patterns for headings set in the body font:
_HEADING_PATTERNS = [
    re.compile(r"^[IVXLC]+\.\s+\S"),                      # "I. Title"
    re.compile(r"^[A-Z]\.\s+\S"),                         # "A. Title"
    re.compile(r"^\d+\.\d+(?:\.\d+)*\.?\s+[A-Z]"),        # "2.1 Title"
    re.compile(r"^[A-Z][A-Z \-:,]{4,}$"),                 # "ALL CAPS TITLE"
]


def body_font_size(doc: LoadedDocument) -> float:
    """Modal font size across the paper = the body text size."""
    sizes = Counter(round(line.size, 1) for page in doc.pages for line in page)
    return sizes.most_common(1)[0][0] if sizes else 10.0


def is_section_heading(line: Line, body_size: float, page_number: int) -> bool:
    """True if a line looks like a section or subsection heading."""
    text = line.text.strip()
    if not (5 <= len(text) <= 90):
        return False
    if text.lower().startswith(_CAPTION_PREFIXES):
        return False

    # Regex fallbacks (applies everywhere, including page 1).
    if any(pattern.match(text) for pattern in _HEADING_PATTERNS):
        return True
    # Single-level numbered heading "2. Methods": must be short and must
    # not end mid-sentence. Excludes list items ("2. Our ... equivalent to")
    # and line numbers in preprint styles ("21 Bohmian mechanics ...").
    if re.match(r"^\d+\.\s+[A-Z]", text):
        return len(text.split()) <= 8 and not _ENDS_MID_SENTENCE.search(text)

    # Font-based detection — skipped on page 1, where titles/author
    # lists/abstract headers are also large or bold.
    if page_number == 1:
        return False
    n_words = len(text.split())
    ends_like_sentence = text.endswith((".", ",", ";", ":"))
    if line.size >= body_size * BODY_SIZE_FACTOR and n_words <= MAX_HEADING_WORDS \
            and not ends_like_sentence:
        return True
    # Bold alone is weak evidence (inline emphasis is often bold), so it
    # only counts for short Capitalized prose without math symbols.
    if line.bold and 2 <= n_words <= 8 and not ends_like_sentence \
            and text[0].isupper() and not any(c in _MATH_CHARS for c in text):
        return True
    return False


def clean_line(line: str) -> str:
    """Collapse internal whitespace, drop ligature debris."""
    return " ".join(line.split())


@dataclass
class Chunk:
    """One retrievable piece of a paper, with provenance metadata."""

    chunk_id: str        # deterministic, e.g. "2304.12345:07"
    paper_id: str
    title: str
    section: str
    text: str
    page: int            # approximate 1-based page where the chunk starts
    index: int           # chunk order within the paper


def _flush(
    buffer_lines: list[str],
    paper_id: str,
    title: str,
    section: str,
    page: int,
    index: int,
) -> Chunk | None:
    """Turn the accumulated lines into one Chunk (None if too small)."""
    text = "\n".join(clean_line(line) for line in buffer_lines if line.strip())
    if len(text) < MIN_CHUNK_CHARS:
        return None
    return Chunk(
        chunk_id=f"{paper_id}:{index:02d}",
        paper_id=paper_id,
        title=title,
        section=section,
        text=text,
        page=page,
        index=index,
    )


def chunk_document(doc: LoadedDocument) -> list[Chunk]:
    """Chunk one loaded paper into section-aware pieces."""
    chunks: list[Chunk] = []
    body_size = body_font_size(doc)
    section = "Preamble"
    buffer: list[str] = []
    chunk_start_page = 1

    for page_number, page in enumerate(doc.pages, start=1):
        for line in page:
            if not line.text.strip():
                continue
            # References/Bibliography is a citation list, not evidence: stop
            # chunking there regardless of heading typography. Reference
            # entries mention every topic word and pollute retrieval.
            if re.match(r"^(references|bibliography)\s*$",
                        line.text.strip(), re.IGNORECASE):
                break
            if is_section_heading(line, body_size, page_number):
                # Heading: close the current chunk, start a new section.
                chunk = _flush(buffer, doc.paper_id, doc.title, section,
                               chunk_start_page, len(chunks))
                if chunk:
                    chunks.append(chunk)
                buffer = []
                section = clean_line(line.text)
                chunk_start_page = page_number
                continue
            if not buffer:
                chunk_start_page = page_number
            buffer.append(line.text)

            if sum(len(text) for text in buffer) >= TARGET_CHUNK_CHARS:
                chunk = _flush(buffer, doc.paper_id, doc.title, section,
                               chunk_start_page, len(chunks))
                if chunk:
                    chunks.append(chunk)
                buffer = []

    chunk = _flush(buffer, doc.paper_id, doc.title, section,
                   chunk_start_page, len(chunks))
    if chunk:
        chunks.append(chunk)

    return chunks


def chunk_all_documents(docs: list[LoadedDocument]) -> list[Chunk]:
    """Chunk every loaded document into one flat list."""
    all_chunks: list[Chunk] = []
    for doc in docs:
        chunks = chunk_document(doc)
        print(f"[chunker] {doc.paper_id}: {len(chunks)} chunks "
              f"({len(doc.text):,} chars, {doc.n_pages} pages)")
        all_chunks.extend(chunks)
    return all_chunks


if __name__ == "__main__":
    from src.documents.loader import load_available_papers

    chunks = chunk_all_documents(load_available_papers())
    print(f"\ntotal chunks: {len(chunks)}")
    sections: dict[str, int] = {}
    for chunk in chunks:
        sections[chunk.section] = sections.get(chunk.section, 0) + 1
    print("detected sections (top 20):")
    for section, count in sorted(sections.items(), key=lambda kv: -kv[1])[:20]:
        print(f"  x{count:3d}  {section[:60]}")
