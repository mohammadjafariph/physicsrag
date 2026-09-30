"""Chunker heading-detection tests on synthetic documents."""

from src.documents.chunker import chunk_document, is_section_heading
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