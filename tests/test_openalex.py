"""OpenAlex fallback tests: parsing, inverted-index abstracts, integration."""

from types import SimpleNamespace
from curl_cffi import CurlError

from src.search import arxiv_search, openalex
from src.search.openalex import (
    extract_arxiv_id,
    parse_work,
    reconstruct_abstract,
    search_openalex,
)


def test_reconstruct_abstract_orders_by_position():
    inverted = {"Entangled": [2], "We": [0], "study": [1], "states": [3]}
    assert reconstruct_abstract(inverted) == "We study Entangled states"


def test_reconstruct_abstract_empty():
    assert reconstruct_abstract({}) == ""
    assert reconstruct_abstract(None) == ""


def test_extract_arxiv_id_from_landing_urls():
    assert extract_arxiv_id(["https://arxiv.org/abs/2304.12345v2"]) == "2304.12345"
    assert extract_arxiv_id(["https://arxiv.org/pdf/2304.12345"]) == "2304.12345"
    assert extract_arxiv_id(["", None, "https://doi.org/10.1000/x"]) == ""
    assert extract_arxiv_id([]) == ""


WORK_WITH_ARXIV = {
    "display_name": "Entanglement scaling",
    "publication_date": "2023-04-12",
    "abstract_inverted_index": {"We": [0], "study": [1], "scaling": [2]},
    "authorships": [{"author": {"display_name": "A Author"}},
                    {"author": {"display_name": "B Author"}}],
    "locations": [{"landing_page_url": "https://arxiv.org/abs/2304.12345v3"},
                  {"landing_page_url": "https://example.com/paper"}],
}

WORK_PUBLISHER_ONLY = {
    "display_name": "Not on arXiv",
    "locations": [{"landing_page_url": "https://doi.org/10.1000/x"}],
    "abstract_inverted_index": {"text": [0]},
}


def test_parse_work_builds_paper_with_arxiv_key():
    paper = parse_work(WORK_WITH_ARXIV)
    assert paper is not None
    assert paper.paper_id == "2304.12345"
    assert paper.title == "Entanglement scaling"
    assert paper.authors == ["A Author", "B Author"]
    assert paper.abstract == "We study scaling"
    assert paper.published == "2023-04-12" and paper.updated == "2023-04-12"
    assert paper.pdf_url == "https://arxiv.org/pdf/2304.12345"


def test_parse_work_skips_publisher_only_records():
    assert parse_work(WORK_PUBLISHER_ONLY) is None
    assert parse_work({**WORK_WITH_ARXIV, "abstract_inverted_index": None}) is None


class FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


def test_search_openalex_dedupes_and_filters(monkeypatch):
    payload = {"results": [WORK_WITH_ARXIV, WORK_PUBLISHER_ONLY,
                           dict(WORK_WITH_ARXIV)]}  # duplicate id
    monkeypatch.setattr(openalex, "requests",
                        SimpleNamespace(get=lambda url, headers=None, timeout=None: FakeResponse(payload)))
    papers = search_openalex("entanglement", max_results=10)
    assert [p.paper_id for p in papers] == ["2304.12345"]


def test_search_papers_falls_back_when_arxiv_down(monkeypatch):
    # arXiv: every request raises (retries exhausted -> re-raise)
    def arxiv_fail(url, headers=None, timeout=None):
        raise CurlError("HTTP Error 429")
    monkeypatch.setattr(arxiv_search, "requests", SimpleNamespace(get=arxiv_fail))
    monkeypatch.setattr(arxiv_search.time, "sleep", lambda s: None)
    # OpenAlex: healthy
    payload = {"results": [WORK_WITH_ARXIV]}
    monkeypatch.setattr(openalex, "requests",
                        SimpleNamespace(get=lambda url, headers=None, timeout=None: FakeResponse(payload)))

    papers = arxiv_search.search_papers(["entanglement scaling"], results_per_query=5)
    assert len(papers) == 1
    assert papers[0].paper_id == "2304.12345"


def test_search_papers_skips_when_both_down(monkeypatch):
    monkeypatch.setattr(arxiv_search, "requests",
                        SimpleNamespace(get=lambda url, headers=None, timeout=None: (_ for _ in ()).throw(CurlError("429"))))
    monkeypatch.setattr(arxiv_search.time, "sleep", lambda s: None)
    def openalex_fail(url, headers=None, timeout=None):
        raise CurlError("connection refused")
    monkeypatch.setattr(openalex, "requests", SimpleNamespace(get=openalex_fail))

    assert arxiv_search.search_papers(["entanglement"], results_per_query=5) == []