"""
Stage 4 — paper downloader.

Takes Paper objects from Stage 3 and persists them to disk:

    list[Paper]
        -> data/papers/{paper_id}.pdf
        -> data/metadata/{paper_id}.json
        -> list[DownloadResult]

Rules:
- A paper is downloaded at most once (existing valid PDFs are skipped).
- A PDF is only saved if it validates (size + %PDF magic bytes).
- Metadata JSON is always written, so every PDF has matching metadata.
"""

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

# Same reason as Stage 3: Python's TLS fingerprint gets 406 from arXiv.
from curl_cffi import requests

from config import METADATA_DIR, PAPERS_DIR
from src.net import NETWORK_ERRORS
from src.search.paper import Paper

ARXIV_PDF_URL = "https://arxiv.org/pdf/{paper_id}"

# arXiv usage guidelines: ~1 request every 3 seconds, including PDFs.
REQUEST_DELAY_SECONDS = 3.0

REQUEST_TIMEOUT_SECONDS = 60

REQUEST_HEADERS = {"User-Agent": "PhysicsRAG/0.1 (autonomous literature research)"}

PDF_MAGIC = b"%PDF"
MIN_PDF_SIZE_BYTES = 10_000  # real papers are always larger than this


@dataclass
class DownloadResult:
    """What happened to one paper during a download run."""

    paper_id: str
    status: str                # "downloaded" | "cached" | "failed"
    pdf_path: str | None = None
    metadata_path: str | None = None
    error: str | None = None


def pdf_path_for(paper_id: str) -> Path:
    """Local PDF path for an arXiv ID, e.g. data/papers/2304.12345.pdf."""
    return PAPERS_DIR / f"{paper_id}.pdf"


def metadata_path_for(paper_id: str) -> Path:
    """Local metadata path, e.g. data/metadata/2304.12345.json."""
    return METADATA_DIR / f"{paper_id}.json"


def save_metadata(paper: Paper) -> Path:
    """Write the Paper's metadata as JSON next to its PDF."""
    path = metadata_path_for(paper.paper_id)
    path.write_text(
        json.dumps(asdict(paper), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return path


def is_valid_pdf(path: Path) -> bool:
    """True if the file exists, has plausible size, and starts with %PDF."""
    if not path.exists():
        return False
    if path.stat().st_size < MIN_PDF_SIZE_BYTES:
        return False
    with path.open("rb") as file:
        return file.read(5).startswith(PDF_MAGIC)


def download_paper(
    paper: Paper,
    request_delay: float = REQUEST_DELAY_SECONDS,
) -> DownloadResult:
    """Download one paper (or skip it if a valid PDF already exists)."""
    target_pdf = pdf_path_for(paper.paper_id)
    metadata_path = save_metadata(paper)  # cheap; keeps metadata always in sync

    if is_valid_pdf(target_pdf):
        return DownloadResult(
            paper_id=paper.paper_id,
            status="cached",
            pdf_path=str(target_pdf),
            metadata_path=str(metadata_path),
        )

    # Pause before every network request: arXiv rate limits PDF fetches too.
    time.sleep(request_delay)

    try:
        url = ARXIV_PDF_URL.format(paper_id=paper.paper_id)
        response = requests.get(
            url,
            headers=REQUEST_HEADERS,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
    except NETWORK_ERRORS as error:
        return DownloadResult(
            paper_id=paper.paper_id,
            status="failed",
            metadata_path=str(metadata_path),
            error=f"{type(error).__name__}: {error}",
        )

    content = response.content
    if not content.startswith(PDF_MAGIC):
        # 200 but not a PDF: arXiv sometimes serves HTML error pages.
        # Never save a broken file — report failure instead.
        return DownloadResult(
            paper_id=paper.paper_id,
            status="failed",
            metadata_path=str(metadata_path),
            error=f"response is not a PDF (starts with {content[:15]!r})",
        )

    target_pdf.write_bytes(content)
    return DownloadResult(
        paper_id=paper.paper_id,
        status="downloaded",
        pdf_path=str(target_pdf),
        metadata_path=str(metadata_path),
    )


def download_papers(
    papers: list[Paper],
    request_delay: float = REQUEST_DELAY_SECONDS,
) -> list[DownloadResult]:
    """Download a batch of papers, skipping ones already on disk."""
    results: list[DownloadResult] = []
    for paper in papers:
        result = download_paper(paper, request_delay=request_delay)
        print(f"[{result.status:>10}] {paper.paper_id}  {paper.title[:60]}")
        results.append(result)
    return results


if __name__ == "__main__":
    from src.search.arxiv_search import search_papers

    demo_queries = [
        "measurement-induced phase transition entanglement entropy scaling",
    ]
    papers = search_papers(demo_queries, results_per_query=3)
    print(f"\ndownloading {len(papers)} papers...\n")
    results = download_papers(papers)

    downloaded = sum(1 for r in results if r.status == "downloaded")
    cached = sum(1 for r in results if r.status == "cached")
    failed = sum(1 for r in results if r.status == "failed")
    print(f"\n=== done: {downloaded} downloaded, {cached} cached, {failed} failed ===")
