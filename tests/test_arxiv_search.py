"""arXiv search tests: retry-with-backoff on transient failures (mocked)."""

import xml.etree.ElementTree as ET

from curl_cffi import CurlError

from src.search import arxiv_search
from src.search.arxiv_search import search_arxiv

ATOM = """<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/2304.12345v1</id>
    <title>Test Paper</title>
    <summary>An abstract about entanglement.</summary>
    <author><name>A Author</name></author>
    <category term="quant-ph"/>
    <published>2023-04-12T00:00:00Z</published>
    <updated>2023-04-12T00:00:00Z</updated>
  </entry>
</feed>"""


class FakeResponse:
    def __init__(self, content=ATOM.encode(), status=200):
        self.content = content
        self.status_code = status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise CurlError(f"HTTP Error {self.status_code}")


def fake_get_factory(outcomes, calls):
    """requests.get stand-in: pops one outcome per call."""
    def fake_get(url, headers=None, timeout=None):
        calls.append(url)
        outcome = outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome
    return fake_get


def test_retry_recovers_after_transient_failures(monkeypatch):
    calls = []
    outcomes = [CurlError("HTTP Error 429"), CurlError("curl: (28) timeout"),
                FakeResponse()]
    monkeypatch.setattr(arxiv_search.requests, "get", fake_get_factory(outcomes, calls))
    monkeypatch.setattr(arxiv_search.time, "sleep", lambda s: None)

    papers = search_arxiv("entanglement", retry_delays=(0.0, 0.0))
    assert len(papers) == 1 and papers[0].paper_id == "2304.12345"
    assert len(calls) == 3  # two failures, then success


def test_retry_reraises_after_exhaustion(monkeypatch):
    calls = []
    outcomes = [CurlError("HTTP Error 429") for _ in range(10)]
    monkeypatch.setattr(arxiv_search.requests, "get", fake_get_factory(outcomes, calls))
    monkeypatch.setattr(arxiv_search.time, "sleep", lambda s: None)

    try:
        search_arxiv("entanglement", retry_delays=(0.0, 0.0))
        raised = False
    except CurlError:
        raised = True
    assert raised
    assert len(calls) == 3  # 1 + 2 retries


def test_first_try_success_does_not_retry(monkeypatch):
    calls = []
    outcomes = [FakeResponse()]
    monkeypatch.setattr(arxiv_search.requests, "get", fake_get_factory(outcomes, calls))
    monkeypatch.setattr(arxiv_search.time, "sleep", lambda s: None)

    papers = search_arxiv("entanglement", retry_delays=(5.0, 15.0))
    assert len(papers) == 1
    assert len(calls) == 1


def test_retry_delays_are_progressive():
    assert arxiv_search.SEARCH_RETRY_DELAYS == (5.0, 15.0, 30.0)
    assert arxiv_search.SEARCH_MAX_ATTEMPTS == len(arxiv_search.SEARCH_RETRY_DELAYS) + 1