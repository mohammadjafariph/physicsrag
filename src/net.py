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
