"""Citation parsing tests (no network)."""

from src.retrieval.retriever import Evidence
from src.service.chat_service import parse_citations


def evidence(n):
    return Evidence(
        chunk_id=f"0000.0000{n}:01", paper_id=f"0000.0000{n}",
        title=f"Paper {n}", section="Introduction", page=1, text="...",
    )


def test_citations_parsed_in_first_appearance_order():
    answer = "The entropy scales logarithmically [2], consistent with [1][2]."
    citations = parse_citations(answer, [evidence(1), evidence(2), evidence(3)])
    assert [c["n"] for c in citations] == [2, 1]
    assert citations[0]["paper_id"] == "0000.00002"


def test_out_of_range_citations_ignored():
    answer = "Claim [7] beyond evidence."
    citations = parse_citations(answer, [evidence(1)])
    assert citations == []


def test_citation_fields_carry_provenance():
    citations = parse_citations("x [1]", [evidence(4)])
    c = citations[0]
    assert c["chunk_id"] == "0000.00004:01"
    assert c["arxiv_url"] == "https://arxiv.org/abs/0000.00004"
    assert c["title"] == "Paper 4"