"""The Paper dataclass — the metadata record for one arXiv paper.

This is a *metadata-only* object: it carries no full text. The PDF body
arrives in Stage 4 (downloader) and Stage 5 (loader).
"""

from dataclasses import dataclass


@dataclass
class Paper:
    """Metadata for one arXiv paper.

    paper_id is the bare arXiv identifier without version suffix
    (e.g. "2304.12345"). It is the unique key used everywhere:
    deduplication (Stage 3), filenames (Stage 4), provenance (Stage 17).
    """

    paper_id: str
    title: str
    authors: list[str]
    abstract: str
    published: str          # ISO date, e.g. "2023-04-26"
    updated: str            # ISO date
    categories: list[str]   # arXiv categories, e.g. ["quant-ph", "cond-mat.str-el"]
    arxiv_url: str          # https://arxiv.org/abs/2304.12345
    pdf_url: str            # https://arxiv.org/pdf/2304.12345
