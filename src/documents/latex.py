"""
Stage 5b — LaTeX source extraction (equation-aware loading).

arXiv PDFs embed equations as rendered glyphs, so the PDF text layer
(Stage 5) mangles complex display math (fractions, matrices, stacked
symbols). arXiv also serves the authors' LaTeX source at the e-print
endpoint, where equations arrive as real LaTeX:

    https://arxiv.org/e-print/{paper_id}   (gzip'd tar, usually)

    list[Equation {latex, context}]
        <- fetch_source(paper_id) + extract_equations(tex)

`context` is the prose paragraph immediately before the equation; the
chunker uses it to attach each equation to the chunk that discusses it.

The module is best-effort by design: any failure (no source, unusual
packaging, parse problems) raises or returns empty and the pipeline
falls back to PDF-only chunking. Results are cached in data/latex/.
"""

import gzip
import io
import json
import re
import tarfile
from dataclasses import asdict, dataclass
from pathlib import Path

from curl_cffi import requests

from config import LATEX_DIR
from src.net import NETWORK_ERRORS

ARXIV_EPRINT_URL = "https://arxiv.org/e-print/{paper_id}"

# arXiv usage guidelines: ~1 request every 3 seconds, same as PDFs.
REQUEST_DELAY_SECONDS = 3.0
REQUEST_TIMEOUT_SECONDS = 60
REQUEST_HEADERS = {"User-Agent": "PhysicsRAG/0.1 (autonomous literature research)"}

MAX_TEX_BYTES = 20_000_000      # sanity cap: a single .tex is never bigger
MAX_INPUT_DEPTH = 10            # \input resolution depth guard
MAX_EQUATIONS_PER_PAPER = 200   # sanity cap on extraction

# Display-math environments to extract (starred variants included).
MATH_ENVIRONMENTS = (
    "equation", "align", "gather", "multline", "eqnarray", "displaymath",
)

_ENV_ALTS = "|".join(re.escape(env) for env in MATH_ENVIRONMENTS)
_ENV_RE = re.compile(
    r"\\begin\{(" + _ENV_ALTS + r")\}(.*?)\\end\{\1\}", re.DOTALL,
)
_BRACKET_RE = re.compile(r"\\\[(.*?)\\\]", re.DOTALL)
_DOLLAR_DOLLAR_RE = re.compile(r"\$\$(.*?)\$\$", re.DOTALL)
_INPUT_RE = re.compile(r"\\(?:input|include)\{([^}]+)\}")


@dataclass
class Equation:
    """One display equation and the prose that introduces it."""

    latex: str   # cleaned LaTeX body, no surrounding \begin{equation}
    context: str  # prose paragraph immediately before the equation


# ---------------------------------------------------------------- fetch ----

def fetch_source(paper_id: str, request_delay: float = REQUEST_DELAY_SECONDS):
    """Raw bytes of the e-print bundle, or None if arXiv has no source."""
    import time

    time.sleep(request_delay)  # arXiv rate limits e-print fetches too
    try:
        response = requests.get(
            ARXIV_EPRINT_URL.format(paper_id=paper_id),
            headers=REQUEST_HEADERS,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
    except NETWORK_ERRORS:
        return None
    content = response.content
    if content[:15].lstrip().lower().startswith(b"<!doctype"):  # HTML error page
        return None
    return content


def extract_tex_files(content: bytes) -> dict[str, str]:
    """Unpack an e-print bundle -> {filename: text}.

    arXiv serves gzip'd tarballs (most common), plain tarballs, gzip'd
    single files, and (rarely) raw .tex. All four land here.
    """
    if not content:
        return {}

    # Case 1: tarball (r:* auto-detects gz/bz2/xz/plain).
    try:
        with tarfile.open(fileobj=io.BytesIO(content), mode="r:*") as tar:
            files: dict[str, str] = {}
            for member in tar.getmembers():
                if not member.isfile():
                    continue
                name = Path(member.name).name  # flatten directories
                if not name.endswith(".tex"):
                    continue
                data = tar.extractfile(member)
                if data is None:
                    continue
                files[name] = _decode(data.read(MAX_TEX_BYTES))
            if files:
                return files
    except (tarfile.TarError, OSError):
        pass

    # Case 2: gzip'd single file.
    if content[:2] == b"\x1f\x8b":
        try:
            return {"paper.tex": _decode(gzip.decompress(content))}
        except OSError:
            return {}

    # Case 3: raw .tex.
    if b"\\documentclass" in content or b"\\begin{document}" in content:
        return {"paper.tex": _decode(content)}
    return {}


def _decode(data: bytes) -> str:
    """TeX files are ASCII-ish; utf-8 first, latin-1 never fails."""
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return data.decode("latin-1")


def main_tex(files: dict[str, str]) -> str:
    """The file with \\documentclass, with \\input/\\include files inlined."""
    main_name = next(
        (name for name, text in files.items() if "\\documentclass" in text),
        None,
    )
    if main_name is None:
        # No documentclass (a section file?): concatenate everything.
        return "\n".join(files.values())

    tex = files[main_name]

    def resolve(match: re.Match, depth: int = 0) -> str:
        if depth >= MAX_INPUT_DEPTH:
            return ""
        name = Path(match.group(1).strip()).name
        if not name.endswith(".tex"):
            name += ".tex"
        return files.get(name, "")

    for _ in range(MAX_INPUT_DEPTH):  # inputs may contain inputs
        resolved = _INPUT_RE.sub(resolve, tex)
        if resolved == tex:
            break
        tex = resolved
    return tex


# -------------------------------------------------------------- cleanup ----

def strip_comments(tex: str) -> str:
    """Drop % comments (keeping \\%, the escaped percent sign)."""
    lines = []
    for line in tex.split("\n"):
        out, i = [], 0
        while i < len(line):
            char = line[i]
            if char == "\\" and i + 1 < len(line):  # escaped char: keep both
                out.append(line[i:i + 2])
                i += 2
                continue
            if char == "%":
                break
            out.append(char)
            i += 1
        lines.append("".join(out))
    return "\n".join(lines)


def body_text(tex: str) -> str:
    """Preamble, comments and verbatim blocks removed, document body kept."""
    tex = strip_comments(tex)
    begin = tex.find("\\begin{document}")
    if begin != -1:
        tex = tex[begin + len("\\begin{document}"):]
    end = tex.find("\\end{document}")
    if end != -1:
        tex = tex[:end]
    tex = re.sub(r"\\begin\{verbatim\}.*?\\end\{verbatim\}", " ", tex, flags=re.DOTALL)
    return tex


# ----------------------------------------------------------- extraction ----

def _clean_equation(latex: str) -> str:
    """Collapse whitespace; equations become single logical lines."""
    return " ".join(latex.split()).strip()


def _clean_prose(text: str) -> str:
    """Loose cleanup of a context paragraph (inline math is kept as-is)."""
    text = re.sub(r"\\[a-zA-Z]+\*?(\[[^\]]*\])?", " ", text)  # simple commands
    text = re.sub(r"[{}]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _context_before(tex: str, start: int) -> str:
    """Prose paragraph immediately preceding tex[start]."""
    before = tex[:start]
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", before) if p.strip()]
    if not paragraphs:
        return ""
    return _clean_prose(paragraphs[-1])[-600:]  # last paragraph, capped


def extract_equations(tex: str) -> list[Equation]:
    """Display equations with their preceding prose context, in order."""
    tex = body_text(tex)
    matches = sorted(
        list(_ENV_RE.finditer(tex))
        + list(_BRACKET_RE.finditer(tex))
        + list(_DOLLAR_DOLLAR_RE.finditer(tex)),
        key=lambda m: m.start(),
    )

    equations: list[Equation] = []
    seen: set[str] = set()
    cursor = 0  # skip overlaps (e.g. $$..$$ inside an align block)
    for match in matches:
        if match.start() < cursor:
            continue
        cursor = match.end()
        latex = _clean_equation(match.groups()[-1])
        if not latex or latex in seen:
            continue
        seen.add(latex)
        equations.append(Equation(
            latex=latex,
            context=_context_before(tex, match.start()),
        ))
        if len(equations) >= MAX_EQUATIONS_PER_PAPER:
            break
    return equations


# ------------------------------------------------------------ pipeline ----

def equations_for_paper(
    paper_id: str,
    use_cache: bool = True,
    request_delay: float = REQUEST_DELAY_SECONDS,
) -> list[Equation]:
    """Equations for one arXiv ID: cache -> e-print fetch -> extract.

    Returns [] when there is no source or nothing extractable; callers
    treat that as 'PDF-only chunking' and carry on.
    """
    cache = LATEX_DIR / f"{paper_id}.json"
    if use_cache and cache.exists():
        return _equations_from_json(cache.read_text(encoding="utf-8"))

    content = fetch_source(paper_id, request_delay=request_delay)
    if content is None:
        return []
    files = extract_tex_files(content)
    if not files:
        return []
    equations = extract_equations(main_tex(files))

    cache.write_text(
        json.dumps([asdict(eq) for eq in equations], ensure_ascii=False),
        encoding="utf-8",
    )
    return equations


def _equations_from_json(raw: str) -> list[Equation]:
    return [Equation(**item) for item in json.loads(raw)]


if __name__ == "__main__":
    import sys

    paper_id = sys.argv[1] if len(sys.argv) > 1 else "2304.12345"
    equations = equations_for_paper(paper_id, use_cache=False)
    print(f"[latex] {paper_id}: {len(equations)} display equations")
    for eq in equations[:10]:
        print(f"\n  context: {eq.context[:120]}")
        print(f"  latex:   {eq.latex[:120]}")