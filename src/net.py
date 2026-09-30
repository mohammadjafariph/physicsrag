"""
curl_cffi compatibility shim (the arXiv HTTP client).

curl_cffi is required for its curl TLS fingerprint (arXiv's edge 406s
Python's own stack), but its exception names moved across versions:

- `curl_cffi.requests.RequestException` never existed at module level in
  modern versions (0.7+) — referencing it from an `except` clause raises
  AttributeError and *crashes the caller while handling a real error*.
- In modern versions the raised class is `RequestsError`, living in
  `curl_cffi.requests.errors`; it subclasses `CurlError`.

`CurlError` is the stable base class every supported version raises, so
call sites catch `NETWORK_ERRORS` and never import exception names
directly from `curl_cffi.requests`.
"""

from curl_cffi import CurlError

try:
    from curl_cffi.requests.errors import RequestsError
except ImportError:  # very old curl_cffi
    RequestsError = CurlError  # type: ignore[assignment, misc]

# Every exception the HTTP layer can raise on network/HTTP failures.
NETWORK_ERRORS: tuple[type[Exception], ...] = (CurlError,)

# Readable alias for call sites (same object as the modern RequestsError).
RequestException = RequestsError


class RetrievalError(RuntimeError):
    """Raised when paper retrieval is impossible: every search query
    failed on arXiv (after retries) AND on the OpenAlex fallback. Run
    loops catch this and stop the cycles — continuing produces a
    garbage analysis built on zero evidence."""


# ---- shared arXiv throttle -------------------------------------------------
#
# Every arXiv touchpoint (search API, PDF downloads, e-print LaTeX fetches)
# used to throttle ITSELF (3s between its own requests), but a research
# cycle chains all three back-to-back — a download can land <1s after a
# search retry, and a new run's first query right after the previous run's
# tail. That burst pattern is what trips arXiv's rate limiter, which then
# answers 429 or tarpits connections (curl 28, 0 bytes). One process-wide
# throttle fixes the stream: minimum spacing between ANY two arXiv requests
# plus a failure-driven cooldown (a 429 means "slow down" for a while).

import threading
import time as _time

ARXIV_MIN_INTERVAL = 3.0        # arXiv usage guidelines: ~1 request / 3s
ARXIV_429_COOLDOWN = 45.0       # a 429 demands a real pause
ARXIV_TIMEOUT_COOLDOWN = 20.0   # stalled/tarpitted connection: shorter pause

_arxiv_lock = threading.Lock()
_arxiv_last_request = 0.0       # monotonic time of the reserved request slot
_arxiv_cooldown_until = 0.0     # monotonic time until which requests wait


def arxiv_pause(min_interval: float = ARXIV_MIN_INTERVAL) -> float:
    """Reserve the next arXiv request slot; return how long to sleep first.

    Enforces (a) >= min_interval since the previous reserved slot and
    (b) any active failure cooldown. Thread-safe: concurrent callers get
    consecutive slots instead of stampeding.
    """
    global _arxiv_last_request
    with _arxiv_lock:
        now = _time.monotonic()
        earliest = max(_arxiv_last_request + min_interval, _arxiv_cooldown_until)
        wait = max(0.0, earliest - now)
        _arxiv_last_request = now + wait  # reserve this slot
    return wait


def arxiv_register_failure(error: Exception) -> None:
    """Extend the shared cooldown based on what arXiv did to us."""
    global _arxiv_cooldown_until
    text = str(error)
    penalty = ARXIV_429_COOLDOWN if "429" in text else ARXIV_TIMEOUT_COOLDOWN
    with _arxiv_lock:
        _arxiv_cooldown_until = max(
            _arxiv_cooldown_until, _time.monotonic() + penalty
        )


def arxiv_classify_failure(error: Exception) -> str:
    """"rate_limited" (HTTP 429) or "timeout" (stalled connection)."""
    return "rate_limited" if "429" in str(error) else "timeout"
