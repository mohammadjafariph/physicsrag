"""Loader tests: PDF glyph-debris filtering and chunk metadata plumbing."""

from src.documents.chunker import Chunk
from src.documents.loader import is_glyph_debris
from src.database.vector_db import VectorStore


def test_glyph_debris_dropped():
    assert is_glyph_debris("ρ∂ψ√")           # symbol soup
    assert is_glyph_debris("12")             # bare page number
    assert is_glyph_debris("···")            # ellipsis debris


def test_real_lines_kept():
    assert not is_glyph_debris("The entanglement entropy scales logarithmically.")
    assert not is_glyph_debris("H = sin(x) + cos(y)")   # named functions survive
    assert not is_glyph_debris("I. Introduction")


def test_chunk_metadata_includes_equations():
    chunk = Chunk(
        chunk_id="2304.12345:00", paper_id="2304.12345", title="T",
        section="S", text="body", page=1, index=0,
        equations=["E = mc^2", "\\hat{H}\\psi = E\\psi"],
    )
    metadata = VectorStore._chunk_metadata(chunk, topic="MIPT")
    assert metadata["equations"] == "E = mc^2\n\n\\hat{H}\\psi = E\\psi"


def test_chunk_metadata_empty_equations():
    chunk = Chunk(chunk_id="1:00", paper_id="1", title="T", section="S",
                  text="body", page=1, index=0)
    assert VectorStore._chunk_metadata(chunk)["equations"] == ""