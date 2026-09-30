"""
Stage 3b — OpenAlex fallback search.

arXiv's API is a single endpoint with no official mirror, and it tarpits
sources that request too fast (stalled connections, 429s). OpenAlex
(https://openalex.org) is a free, key-less academic index that covers
essentially all of arXiv preprints:

    https://api.openalex.org/works?search=...&per-page=N
        -> list[Paper]

Papers are keyed by the arXiv ID scraped from OpenAlex landing-page URLs,
so everything downstream (download, chunk, cite) keeps working unchanged;
the PDF still comes from arXiv itself. Works with no arXiv landing page
(published-only records) are skipped — the pipeline is arXiv-keyed.

Abstracts arrive as an inverted index {word: [positions]}; we rebuild the
plain text from it.
"""

import re
from urllib.parse import quote

from curl_cffi import requests

from src.search.paper import Paper

OPENALEX_API_URL = "https://api.openalex.org/works"

REQUEST_TIMEOUT_SECONDS = 20
REQUEST_HEADERS = {"User-Agent": "PhysicsRAG/0.1 (autonomous literature research)"}

# arXiv landing pages in OpenAlex look like .../abs/2304.12345v2.
_ARXIV_URL_RE = re.compile(r"arxiv\.org/(?:abs|pdf)/([A-Za-z0-9.\-]+)")


def extract_arxiv_id(urls: list[str | None]) -> str:
    """Bare arXiv id from any OpenAlex landing-page URL, '' if none."""
    for url in urls:
        match = _ARXIV_URL_RE.search(url or "")
        if match:
            raw = match.group(1)
            # drop version suffix, same convention as the arXiv loader
            return raw.split("v")[0] if "v" in raw.split("/")[-1] else raw
    return ""


def reconstruct_abstract(inverted: dict | None) -> str:
    """OpenAlex inverted-index abstract -> plain text.

    {word: [0, 4], other: [1]} -> "word other ..." (position order).
    """
    if not inverted:
        return ""
    words: dict[int, str] = {}
    for word, positions in inverted.items():
        for position in positions or []:
            words[position] = word
    return " ".join(words[position] for position in sorted(words))


def parse_work(work: dict) -> Paper | None:
    """One OpenAlex work -> Paper, or None when not an arXiv-keyed paper."""
    landing_urls = [
        location.get("landing_page_url") or ""
        for location in work.get("locations") or []
    ]
    landing_urls.append((work.get("primary_location") or {}).get("landing_page_url") or "")
    paper_id = extract_arxiv_id(landing_urls)
    if not paper_id:
        return None

    abstract = reconstruct_abstract(work.get("abstract_inverted_index") or {})
    if not abstract:
        return None  # no abstract -> useless for search relevance

    authors = [
        (authorship.get("author") or {}).get("display_name") or "Unknown"
        for authorship in work.get("authorships") or []
    ]
    published = work.get("publication_date") or ""
    return Paper(
        paper_id=paper_id,
        title=(work.get("display_name") or "").strip(),
        authors=authors,
        abstract=" ".join(abstract.split()),
        published=published,
        updated=published,   # OpenAlex tracks only one date here
        categories=[],       # concepts are too coarse to be arXiv categories
        arxiv_url=f"https://arxiv.org/abs/{paper_id}",
        pdf_url=f"https://arxiv.org/pdf/{paper_id}",
    )


def search_openalex(query: str, max_results: int = 10) -> list[Paper]:
    """Search OpenAlex; returns arXiv-keyed Papers (may be fewer than asked)."""
    url = (
        f"{OPENALEX_API_URL}"
        f"?search={quote(query)}"
        f"&per-page={min(max_results, 50)}"
        f"&sort=relevance_score:desc"
    )
    response = requests.get(
        url,
        headers=REQUEST_HEADERS,
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    works = response.json().get("results") or []

    papers: list[Paper] = []
    seen: set[str] = set()
    for work in works:
        paper = parse_work(work)
        if paper is not None and paper.paper_id not in seen:
            seen.add(paper.paper_id)
            papers.append(paper)
    return papers
