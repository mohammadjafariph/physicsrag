"""
Stage 3 — arXiv search engine.

Turns a list of search queries into deduplicated Paper metadata objects:

    ["measurement-induced phase transition entanglement entropy", ...]
        -> arXiv API (Atom XML)
        -> list[Paper]

No PDFs are downloaded here; that is Stage 4's job.
"""

import time
import xml.etree.ElementTree as ET
from urllib.parse import quote

# curl_cffi, NOT `requests`: arXiv's edge (Google Frontend) blocks Python's
# TLS fingerprint with HTTP 406, while curl's handshake is accepted. Same
# URL + headers succeed via curl and fail via requests/urllib, so the
# fingerprint is the trigger. curl_cffi speaks with libcurl's TLS stack
# and is a drop-in replacement for requests here.
from curl_cffi import requests

from src.search.paper import Paper
from src.search.rank import STOPWORDS as _STOPWORDS

ARXIV_API_URL = "https://export.arxiv.org/api/query"

# arXiv asks for at most ~1 request every 3 seconds (usage guidelines).
REQUEST_DELAY_SECONDS = 3.0

REQUEST_TIMEOUT_SECONDS = 30

# arXiv's usage guidelines ask for a descriptive User-Agent identifying
# the tool. (Not sufficient on its own — the TLS fingerprint above is.)
REQUEST_HEADERS = {"User-Agent": "PhysicsRAG/0.1 (autonomous literature research)"}

# Atom XML namespace: every tag in the response lives inside this namespace.
ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}


def build_search_url(query: str, max_results: int, sort_by: str = "relevance") -> str:
    """Build the arXiv API URL for one query.

    arXiv has no fuzzy matching: quoting the whole query as a phrase only
    finds abstracts containing that exact word sequence, which almost
    never happens for planner-generated phrases. Instead we split the
    query into individual terms (hyphens become spaces) and require all
    of them with AND:

        "measurement-induced phase transition"
            -> all:measurement AND all:induced AND all:phase AND all:transition
    """
    terms = query.replace("-", " ").split()
    arxiv_query = " AND ".join(f"all:{term}" for term in terms)
    encoded_query = quote(arxiv_query)
    return (
        f"{ARXIV_API_URL}"
        f"?search_query={encoded_query}"
        f"&sortBy={sort_by}"
        f"&sortOrder=descending"
        f"&max_results={max_results}"
    )


def extract_bare_id(entry_id_url: str) -> str:
    """Reduce an entry id URL to the bare arXiv ID.

    "http://arxiv.org/abs/2304.12345v2" -> "2304.12345"
    The version suffix (v2) is dropped: we always track the latest version
    and the bare ID is our stable dedupe/filename key.
    """
    raw_id = entry_id_url.rstrip().split("/abs/")[-1]
    last_segment = raw_id.split("/")[-1]
    return last_segment.split("v")[0] if "v" in last_segment else last_segment


def looks_withdrawn(abstract: str) -> bool:
    """Heuristic: arXiv replaces withdrawn papers' abstracts with a notice."""
    lowered = abstract.lower()
    return "withdrawn" in lowered or "retracted" in lowered


def parse_entry(entry: ET.Element) -> Paper | None:
    """Parse one Atom <entry> into a Paper (None for withdrawn papers)."""
    entry_id_url = entry.findtext("atom:id", "", ATOM_NS)
    title = " ".join(entry.findtext("atom:title", "", ATOM_NS).split())
    abstract = " ".join(entry.findtext("atom:summary", "", ATOM_NS).split())

    if not entry_id_url or looks_withdrawn(abstract):
        return None

    authors = [
        author.findtext("atom:name", "Unknown", ATOM_NS)
        for author in entry.findall("atom:author", ATOM_NS)
    ]
    categories = [
        category.get("term", "")
        for category in entry.findall("atom:category", ATOM_NS)
    ]

    bare_id = extract_bare_id(entry_id_url)
    return Paper(
        paper_id=bare_id,
        title=title,
        authors=authors,
        abstract=abstract,
        published=entry.findtext("atom:published", "", ATOM_NS)[:10],
        updated=entry.findtext("atom:updated", "", ATOM_NS)[:10],
        categories=categories,
        arxiv_url=f"https://arxiv.org/abs/{bare_id}",
        pdf_url=f"https://arxiv.org/pdf/{bare_id}",
    )


def search_arxiv(query: str, max_results: int = 10) -> list[Paper]:
    """Run one query against the arXiv API and return Papers."""
    url = build_search_url(query, max_results=max_results)
    response = requests.get(
        url,
        headers=REQUEST_HEADERS,
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()  # non-200 -> exception with the status code

    root = ET.fromstring(response.content)
    papers = [
        paper
        for entry in root.findall("atom:entry", ATOM_NS)
        if (paper := parse_entry(entry)) is not None
    ]
    return papers


MIN_RELAXED_TERMS = 3
RELAXATION_SCHEDULE = (5, 4, 3)   # kept-term counts tried in order


def search_arxiv_relaxed(
    query: str,
    max_results: int = 10,
    request_delay: float = REQUEST_DELAY_SECONDS,
) -> tuple[list[Paper], str | None]:
    """Search, progressively relaxing over-restrictive queries.

    arXiv has no fuzzy matching, so a planner query with 6-8 AND-ed terms
    often returns nothing. Relaxation strategy:
    1. strip stopwords ("of", "using", ...),
    2. if still empty, retry with fewer kept terms: the first n-1 terms
       (the planner's core concept) PLUS the last term (often the most
       distinctive one, e.g. "... potential Bohmian").
    Every retry sleeps request_delay — arXiv rate limits apply per call.
    """
    terms = [
        term for term in query.replace("-", " ").split()
        if term.lower() not in _STOPWORDS
    ]
    papers = search_arxiv(" ".join(terms), max_results=max_results)
    if papers:
        return papers, None

    relaxed_query = None
    for n_kept in RELAXATION_SCHEDULE:
        if n_kept >= len(terms):
            continue
        kept = terms[: n_kept - 1] + [terms[-1]]
        relaxed_query = " ".join(kept)
        time.sleep(request_delay)  # each retry is its own API call
        papers = search_arxiv(relaxed_query, max_results=max_results)
        if papers:
            return papers, relaxed_query
    return papers, relaxed_query


def search_papers(
    queries: list[str],
    results_per_query: int = 10,
    request_delay: float = REQUEST_DELAY_SECONDS,
) -> list[Paper]:
    """Run several queries and return the merged, deduplicated papers.

    Deduplication: a dict keyed by paper_id keeps the first occurrence
    (highest relevance, because arXiv sorts by relevance) and drops
    repeats from later queries. Insertion order is preserved.
    """
    merged: dict[str, Paper] = {}

    for index, query in enumerate(queries):
        print(f"[search {index + 1}/{len(queries)}] {query}")
        # arXiv rate limit: pause before every request except the first.
        if index > 0:
            time.sleep(request_delay)

        try:
            papers, relaxed_query = search_arxiv_relaxed(
                query, max_results=results_per_query
            )
        except requests.RequestException as error:
            # One failed query must not kill the whole search batch.
            print(f"  query failed, skipping: {error}")
            continue

        if relaxed_query:
            print(f"  [relaxed] -> {relaxed_query!r}")
        for paper in papers:
            merged.setdefault(paper.paper_id, paper)
        print(f"  {len(papers)} results (total unique: {len(merged)})")

    return list(merged.values())


if __name__ == "__main__":
    demo_queries = [
        "measurement-induced phase transition entanglement entropy scaling",
        "random unitary circuits projective measurements critical point",
        "experimental observation measurement-induced transition",
    ]
    found = search_papers(demo_queries, results_per_query=5)
    print(f"\n=== {len(found)} unique papers ===")
    for paper in found:
        print(f"- [{paper.paper_id}] {paper.title} ({paper.published})")
