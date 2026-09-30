"""Stage 5b tests: LaTeX extraction, packaging, and equation contexts."""

import gzip
import io
import json
import tarfile

from src.documents.latex import (
    Equation,
    extract_equations,
    extract_tex_files,
    main_tex,
    strip_comments,
)


SAMPLE_TEX = r"""
\documentclass{article}
\usepackage{amsmath}
\title{Entropy}
\begin{document}
The entanglement entropy of a pure state is the von Neumann entropy
of the reduced density matrix % inline comment
\begin{equation}
S(\rho_A) = -\mathrm{Tr}\,\rho_A \log \rho_A
\end{equation}
which quantifies entanglement across the bipartition.

For two intervals the mutual information is bounded by
\begin{align}
I(A,B) &\geq \frac{c}{3}\log\frac{\ell}{\epsilon} \\
       &\quad + 2\,\gamma
\end{align}
at large central charge.

The free-particle Hamiltonian reads
\[
\hat{H} = -\frac{\hbar^2}{2m}\nabla^2
\]
and evolution follows
$$
\psi(t) = e^{-i\hat{H}t/\hbar}\psi(0)
$$
Escaped percent: 50\% of states. This is prose with the, of, and words.
\end{document}
"""


def test_extract_equations_finds_all_display_math():
    equations = extract_equations(SAMPLE_TEX)
    latex_bodies = [eq.latex for eq in equations]
    assert any("S(\\rho_A)" in body for body in latex_bodies)
    assert any("I(A,B)" in body for body in latex_bodies)
    assert any("\\hat{H} = -\\frac{\\hbar^2}{2m}" in body for body in latex_bodies)
    assert any("e^{-i\\hat{H}t" in body for body in latex_bodies)
    assert len(equations) == 4


def test_context_is_the_prose_paragraph_before():
    equations = extract_equations(SAMPLE_TEX)
    von_neumann = next(eq for eq in equations if "S(\\rho_A)" in eq.latex)
    assert "von Neumann entropy" in von_neumann.context
    mutual = next(eq for eq in equations if "I(A,B)" in eq.latex)
    assert "mutual information" in mutual.context


def test_comments_and_escaped_percent():
    tex = "% a full-line comment\nx = 1 \\% kept % trailing comment\n"
    cleaned = strip_comments(tex)
    assert "kept" in cleaned
    assert "\\%" in cleaned
    assert "trailing comment" not in cleaned
    assert "full-line comment" not in cleaned


def test_inline_comments_do_not_break_equation_extraction():
    equations = extract_equations(SAMPLE_TEX)
    assert len(equations) == 4  # the % inline comment above equation 1


def test_preamble_and_document_markers_removed():
    equations = extract_equations(SAMPLE_TEX)
    all_text = " ".join(eq.context for eq in equations)
    assert "\\documentclass" not in all_text
    assert "amsmath" not in all_text


def test_prose_is_not_mistaken_for_an_equation():
    equations = extract_equations(SAMPLE_TEX)
    joined = " ".join(eq.latex for eq in equations)
    assert "Escaped percent" not in joined


def test_extract_tex_files_tarball():
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        for name, text in [("main.tex", "\\documentclass{article}\\input{sec}"),
                           ("sec/sec.tex", "Section body")]:
            data = text.encode()
            info = tarfile.TarInfo(name=name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    files = extract_tex_files(buffer.getvalue())
    assert set(files) == {"main.tex", "sec.tex"}


def test_extract_tex_files_gzipped_single_file():
    content = gzip.compress(b"\\documentclass{article}\\begin{document}x\\end{document}")
    files = extract_tex_files(content)
    assert list(files) == ["paper.tex"]


def test_main_tex_resolves_input():
    files = {
        "main.tex": "\\documentclass{article}\n\\input{sections}\n\\begin{document}body\\end{document}",
        "sections.tex": "The section text.",
    }
    tex = main_tex(files)
    assert "The section text." in tex
    assert "\\input" not in tex


def test_main_tex_concatenates_without_documentclass():
    files = {"a.tex": "part one", "b.tex": "part two"}
    tex = main_tex(files)
    assert "part one" in tex and "part two" in tex


def test_equation_roundtrip_json():
    equation = Equation(latex="E = mc^2", context="mass energy equivalence")
    raw = json.dumps([{"latex": equation.latex, "context": equation.context}])
    restored = [Equation(**item) for item in json.loads(raw)]
    assert restored == [equation]