"""Shared arXiv throttle: spacing, failure cooldowns, retry integration."""

from types import SimpleNamespace

from curl_cffi import CurlError

import src.net as net
from src.search import arxiv_search

ARXIV_ENTRY_FEED = b"""<feed xmlns="http://www.w3.org/2005/Atom">
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


class FakeClock:
    def __init__(self, start=100.0):
        self.t = start

    def monotonic(self):
        return self.t

    def advance(self, dt):
        self.t += dt


def reset_throttle(monkeypatch, clock):
    net._arxiv_last_request = 0.0
    net._arxiv_cooldown_until = 0.0
    monkeypatch.setattr(net._time, "monotonic", clock.monotonic)


def test_spacing_between_consecutive_slots(monkeypatch):
    clock = FakeClock()
    reset_throttle(monkeypatch, clock)
    w1 = net.arxiv_pause()     # epoch slot long past -> no wait
    w2 = net.arxiv_pause()     # second slot: 3s after the first reservation
    assert w1 == 0.0
    assert w2 == 3.0
    clock.advance(6.0)         # sleep w2, fire request 2, spacing now past
    assert net.arxiv_pause() == 0.0


def test_429_cooldown_is_long(monkeypatch):
    clock = FakeClock()
    reset_throttle(monkeypatch, clock)
    net.arxiv_register_failure(CurlError("HTTP Error 429: "))
    wait = net.arxiv_pause()
    assert 40 <= wait <= 45    # full 45s cooldown still ahead
    assert net.arxiv_classify_failure(CurlError("HTTP Error 429: ")) == "rate_limited"


def test_timeout_cooldown_is_shorter(monkeypatch):
    clock = FakeClock()
    reset_throttle(monkeypatch, clock)
    net.arxiv_register_failure(CurlError("curl: (28) Operation timed out"))
    wait = net.arxiv_pause()
    assert 15 <= wait <= 20
    assert net.arxiv_classify_failure(CurlError("curl: (28) timed out")) == "timeout"


def test_cooldown_takes_max_of_repeated_failures(monkeypatch):
    clock = FakeClock()
    reset_throttle(monkeypatch, clock)
    net.arxiv_register_failure(CurlError("curl: (28) timed out"))
    clock.t += 10
    net.arxiv_register_failure(CurlError("HTTP Error 429"))
    wait = net.arxiv_pause()
    assert 40 <= wait <= 45    # the later, longer cooldown wins


def test_search_retry_sleeps_out_cooldown_before_second_attempt(monkeypatch):
    clock = FakeClock()
    reset_throttle(monkeypatch, clock)

    class FakeResponse:
        def __init__(self, status=200):
            self.content = ARXIV_ENTRY_FEED
            self._status = status

        def raise_for_status(self):
            if self._status >= 400:
                raise CurlError(f"HTTP Error {self._status}: ")

    sleeps = []
    calls = []
    outcomes = [FakeResponse(status=429), FakeResponse()]

    def fake_get(url, headers=None, timeout=None):
        calls.append(url)
        return outcomes.pop(0)

    monkeypatch.setattr(arxiv_search, "requests", SimpleNamespace(get=fake_get))
    monkeypatch.setattr(arxiv_search.time, "sleep", lambda s: (sleeps.append(s), clock.advance(s)))

    papers = arxiv_search.search_arxiv("entanglement", retry_delays=(5.0, 15.0, 30.0))
    assert len(papers) == 1
    assert len(calls) == 2
    # attempt 1: no wait (slot long past) -> not even recorded as a sleep;
    # attempt 2 waits out the FULL 45s cooldown registered by the 429 —
    # retry_delays no longer drive the pacing, the shared throttle does.
    assert len(sleeps) == 1
    assert 40 <= sleeps[0] <= 45