"""Regression tests: network failures must be reported, never crash the pipeline.

The original bug: `except requests.RequestException` referenced an
exception curl_cffi no longer exports, so the error handler itself raised
AttributeError and killed the whole research run mid-cycle.
"""

import pytest

from src.documents import downloader
from src.net import CurlError, NETWORK_ERRORS, RequestException
from src.search import arxiv_search
from src.search.paper import Paper


def test_request_exception_is_subclass_of_curl_error():
    assert issubclass(RequestException, CurlError)
    assert all(issubclass(e, Exception) for e in NETWORK_ERRORS)


def test_network_errors_actually_catch():
    with pytest.raises(NETWORK_ERRORS):
        raise RequestException("simulate arXiv failure")


def _paper(paper_id="2401.12345"):
    return Paper(
        paper_id=paper_id, title="t", authors=[], abstract="a",
        published="2024-01-01", updated="2024-01-01", categories=[],
        arxiv_url=f"https://arxiv.org/abs/{paper_id}",
        pdf_url=f"https://arxiv.org/pdf/{paper_id}",
    )


def test_download_failure_returns_failed_result(tmp_path, monkeypatch):
    """A raising HTTP layer -> DownloadResult(status='failed'), no exception."""
    def boom(url, **kwargs):
        raise RequestException("HTTP 404: bad arXiv id")

    monkeypatch.setattr(downloader, "pdf_path_for", lambda pid: tmp_path / f"{pid}.pdf")
    monkeypatch.setattr(
        downloader, "metadata_path_for", lambda pid: tmp_path / f"{pid}.json"
    )
    monkeypatch.setattr(downloader.requests, "get", boom)
    monkeypatch.setattr(downloader.time, "sleep", lambda s: None)

    result = downloader.download_paper(_paper())
    assert result.status == "failed"
    assert result.pdf_path is None
    assert "HTTP 404" in result.error


def test_search_papers_skips_failing_query(monkeypatch):
    """One dead query must not kill the batch; later queries still count."""
    good = [_paper("1111.22221")]
    calls = {"n": 0}

    def flaky(query, max_results=10, request_delay=3.0):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RequestException("rate limited")
        return good, None

    monkeypatch.setattr(arxiv_search, "search_arxiv_relaxed", flaky)
    monkeypatch.setattr(arxiv_search.time, "sleep", lambda s: None)

    found = arxiv_search.search_papers(["dead query", "good query"], request_delay=0)
    assert calls["n"] == 2
    assert [p.paper_id for p in found] == ["1111.22221"]