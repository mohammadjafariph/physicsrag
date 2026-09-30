"""Chunker heading-detection tests on synthetic documents."""

from src.documents.chunker import (
    chunk_document,
    is_section_heading,
    looks_like_display_math,
    attach_equations,
)
from src.documents.latex import Equation
from src.documents.loader import Line, LoadedDocument


def line(text, size=10.0, bold=False):
    return Line(text=text, size=size, bold=bold)


def make_doc(pages):
    return LoadedDocument(paper_id="0000.00001", title="Test paper", pages=pages)


def body(chars, size=10.0):
    """One long body line (chunks need >= 150 chars to be kept)."""
    return line("word " * (chars // 5), size=size)


def test_font_based_heading_detected_after_page_1():
    doc = make_doc([
        [body(400)],                    # page 1: font detection is off
        [line("Methods", size=12.0), body(400)],
    ])
    chunks = chunk_document(doc)
    assert any(chunk.section == "Methods" for chunk in chunks)


def test_regex_heading_on_page_1():
    doc = make_doc([
        [line("I. Introduction"), body(400)],
        [line("2.1 Apparatus"), body(400)],
    ])
    chunks = chunk_document(doc)
    sections = {chunk.section for chunk in chunks}
    assert "I. Introduction" in sections
    assert "2.1 Apparatus" in sections


def test_references_stops_chunking():
    doc = make_doc([
        [body(400)],
        [line("References"), line("[1] Someone, Some paper, 2020."), line("[2] Another one.")],
    ])
    chunks = chunk_document(doc)
    assert all("Some paper" not in chunk.text for chunk in chunks)


def test_short_chunks_merged_or_dropped():
    doc = make_doc([[line("I. Introduction"), body(60)]])
    chunks = chunk_document(doc)
    assert chunks == []  # 60*5=300 chars? no — body(60) makes "word "*12 = 60 chars


def test_small_bold_line_is_heading():
    heading = line("Data analysis", size=10.0, bold=True)
    body_size = 10.0
    assert is_section_heading(heading, body_size, page_number=2)


def test_caption_is_never_a_heading():
    caption = line("Figure 1: entropy vs time", size=12.0)
    assert not is_section_heading(caption, 10.0, page_number=2)


# ---- Stage 5b: equations ----------------------------------------------------

def test_display_math_detected():
    assert looks_like_display_math("E(ρ,ψ) = -(ℏ2/2m)∇2ψ + V(ρ,ψ)ψ")
    assert looks_like_display_math("S = -Tr ρ log ρ")
    assert looks_like_display_math("⟨ψ|O|ψ⟩ ∝ ∑ λ_i")


def test_prose_is_not_display_math():
    assert not looks_like_display_math(
        "The entropy of the reduced density matrix is given by the trace"
    )
    assert not looks_like_display_math(
        "We can see that the result follows from the previous section."
    )


def test_equation_line_glued_to_previous_chunk():
    # A target-size flush cuts the chunk right before a display equation;
    # the equation must be glued back onto the previous chunk's text.
    first = line("word " * 260)          # ~1300 chars -> triggers a flush
    equation = line("E(ρ,ψ) = -(ℏ2/2m)∇2ψ + V(ρ,ψ)ψ")
    after = line("word " * 40)
    doc = make_doc([[first, equation, after]])
    chunks = chunk_document(doc)
    assert chunks
    assert equation.text in chunks[0].text


def test_equation_after_heading_starts_new_chunk():
    # No glue across a section heading: the equation belongs to the new
    # section's first chunk.
    doc = make_doc([
        [line("word " * 260)],
        [line("Methods", size=12.0), line("H = p^2/2m"), line("word " * 40)],
    ])
    chunks = chunk_document(doc)
    methods = next(c for c in chunks if c.section == "Methods")
    assert "H = p^2/2m" in methods.text.split("\n")[0]


def test_attach_equations_matches_by_context():
    chunks = chunk_document(make_doc([
        [line("I. Entanglement"), line("The entanglement entropy of a pure state and its scaling behavior " * 3)],
        [line("II. Dynamics"), line("The time evolution of the wavefunction under the Hamiltonian operator " * 3)],
    ]))
    equations = [
        Equation(latex="S = -Tr rho log rho",
                 context="The entanglement entropy of a pure state is defined as"),
        Equation(latex="d psi/dt = -i H psi",
                 context="Time evolution of the wavefunction under the Hamiltonian"),
    ]
    attached = attach_equations(chunks, equations)
    assert attached == 2
    assert chunks[0].equations == ["S = -Tr rho log rho"]
    assert chunks[1].equations == ["d psi/dt = -i H psi"]
    assert "S = -Tr rho log rho" in chunks[0].text


def test_attach_equations_drops_unmatchable():
    chunks = chunk_document(make_doc([[line("word " * 60)]]))
    equations = [Equation(latex="x = 1", context="completely unrelated quantum gravity holography")]
    assert attach_equations(chunks, equations) == 0
    assert chunks[0].equations == []