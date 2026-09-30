"""Lexical paper ranking tests."""

from src.search.paper import Paper
from src.search.rank import rank_papers, score_paper


def paper(title, abstract, published="2023-01-01"):
    return Paper(
        paper_id="0000.00001", title=title, authors=[], abstract=abstract,
        published=published, updated=published, categories=[],
        arxiv_url="", pdf_url="",
    )


def test_title_hits_score_higher_than_abstract_hits():
    terms = ["measurement", "transition"]
    in_title = score_paper(paper("measurement-induced transition", "nothing"), terms)
    in_abstract = score_paper(paper("other topic", "measurement transition"), terms)
    assert in_title > in_abstract


def test_rank_papers_keeps_top_limit():
    papers = [
        paper("measurement-induced phase transition in circuits", "measurement transition phase"),
        paper("bohmian mechanics basics", "bohmian"),
        paper("completely unrelated stuff", "unrelated"),
    ]
    top = rank_papers(papers, ["measurement-induced phase transition"], limit=1)
    assert len(top) == 1
    assert "measurement" in top[0].title.lower()


def test_recency_prefers_newer_papers():
    papers = [
        paper("measurement transition", "measurement transition", published="2020-01-01"),
        paper("measurement transition", "measurement transition", published="2025-01-01"),
    ]
    top = rank_papers(papers, ["measurement transition"], limit=1)
    assert top[0].published == "2025-01-01"