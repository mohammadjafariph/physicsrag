"""
Stage 5 — document loader.

Extracts structured text from downloaded PDFs:

    data/papers/{id}.pdf  (+  data/metadata/{id}.json)
        -> LoadedDocument {paper_id, title, pages, full text}

Each page is a list of Line objects carrying FONT information (size, bold)
extracted by PyMuPDF. The chunker (Stage 6) uses this to detect section
headings typographically instead of guessing from text patterns alone.
"""

from dataclasses import dataclass, field
import json
import re
from pathlib import Path

import pymupdf

from config import METADATA_DIR, PAPERS_DIR

BOLD_FLAG = 16  # PyMuPDF span flags bit 4 (2**4) == bold

# Glyph-debris filter: equation glyphs extract as isolated symbols with no
# real word on the line. Prose and named functions (sin, exp, det, ...) keep
# runs of >= 3 letters; debris like 'ρ∂ψ√' and bare page numbers do not.
_WORD_RUN_RE = re.compile(r"[A-Za-z]{3,}")


def is_glyph_debris(text: str) -> bool:
    """True for lines with no run of 3+ letters (symbol soup, page numbers)."""
    return _WORD_RUN_RE.search(text) is None


@dataclass
class Line:
    """One rendered line of a PDF page, with its dominant font traits."""

    text: str
    size: float   # largest span font size on the line (points)
    bold: bool


@dataclass
class LoadedDocument:
    """Text content of one paper, page by page, font-aware."""

    paper_id: str
    title: str
    pages: list[list[Line]]   # pages[i] is page i+1 as a list of Lines
    text: str = field(default="", repr=False)   # all lines joined

    @property
    def n_pages(self) -> int:
        return len(self.pages)


def load_metadata(paper_id: str) -> dict:
    """Read the metadata JSON saved by the downloader."""
    path = METADATA_DIR / f"{paper_id}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def extract_page_lines(page: "pymupdf.Page") -> list[Line]:
    """Extract one page's lines with font size and bold information."""
    lines: list[Line] = []
    page_dict = page.get_text("dict")
    for block in page_dict["blocks"]:
        if block.get("type") != 0:  # 0 = text block (skip images)
            continue
        for raw_line in block["lines"]:
            spans = raw_line.get("spans", [])
            if not spans:
                continue
            text = " ".join(span["text"] for span in spans).strip()
            if not text:
                continue
            if is_glyph_debris(text):
                continue
            size = max(span["size"] for span in spans)
            bold = any(span["flags"] & BOLD_FLAG for span in spans)
            lines.append(Line(text=" ".join(text.split()), size=size, bold=bold))
    return lines


def load_pdf(pdf_path: Path, paper_id: str = "", title: str = "") -> LoadedDocument:
    """Extract text from one PDF file."""
    pages: list[list[Line]] = []
    with pymupdf.open(pdf_path) as doc:
        for page in doc:
            pages.append(extract_page_lines(page))

    full_text = "\n".join(
        line.text for page in pages for line in page
    )
    return LoadedDocument(
        paper_id=paper_id or pdf_path.stem,
        title=title,
        pages=pages,
        text=full_text,
    )


def load_paper(paper_id: str) -> LoadedDocument:
    """Load one paper by its arXiv ID, using saved metadata for the title."""
    metadata = load_metadata(paper_id)
    pdf_path = PAPERS_DIR / f"{paper_id}.pdf"
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not downloaded: {pdf_path}")
    return load_pdf(pdf_path, paper_id=paper_id, title=metadata.get("title", ""))


def load_available_papers() -> list[LoadedDocument]:
    """Load every paper that has both a PDF and metadata on disk."""
    paper_ids = sorted(
        path.stem
        for path in PAPERS_DIR.glob("*.pdf")
        if (METADATA_DIR / f"{path.stem}.json").exists()
    )
    return [load_paper(paper_id) for paper_id in paper_ids]


if __name__ == "__main__":
    docs = load_available_papers()
    print(f"loaded {len(docs)} papers")
    for doc in docs:
        n_lines = sum(len(page) for page in doc.pages)
        print(f"- [{doc.paper_id}] {doc.title[:60]} | {doc.n_pages} pages, "
              f"{n_lines} lines, {len(doc.text):,} chars")
    if docs:
        print("\n--- page 1 first lines with font info ---")
        for line in docs[0].pages[0][:6]:
            print(f"  size={line.size:5.1f} bold={line.bold} | {line.text[:70]}")
