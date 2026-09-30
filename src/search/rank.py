"""
Stage 19a — paper filter/rank (the `filter_and_rank` box of Stage 14).

Search returns papers ranked per-query by arXiv relevance, but after
merging and relaxing queries the merged list contains off-target hits.
Before spending download bandwidth we re-score every paper against the
full plan:

    score = 2 * title-term coverage + 1 * abstract-term coverage
          (+ small recency bonus, newer papers preferred on ties)

and keep only the top `limit`. Purely lexical, instant, no LLM — the
expensive semantic judgment happens later, on downloaded chunks.
"""

from src.search.paper import Paper

# Words that waste query terms / scoring without narrowing the topic.
STOPWORDS = {
    "of", "the", "a", "an", "in", "on", "for", "with", "from", "and",
    "via", "using", "toward", "towards", "into", "by", "at",
}

TITLE_WEIGHT = 2.0
ABSTRACT_WEIGHT = 1.0
RECENCY_BONUS = 0.05      # per year newer than the oldest paper in the batch
RECENCY_CAP = 0.25        # never let recency dominate term coverage


def plan_terms(queries: list[str]) -> list[str]:
    """Distinctive lowercase terms across all plan queries."""
    terms: set[str] = set()
    for query in queries:
        for term in query.replace("-", " ").split():
            lowered = term.lower()
            if len(lowered) >= 3 and lowered not in STOPWORDS:
                terms.add(lowered)
    return sorted(terms)


def score_paper(paper: Paper, terms: list[str]) -> float:
    """Lexical relevance of one paper to the plan's term set."""
    title = paper.title.lower()
    abstract = paper.abstract.lower()
    title_hits = sum(1 for term in terms if term in title)
    abstract_hits = sum(1 for term in terms if term in abstract)
    coverage = (TITLE_WEIGHT * title_hits + ABSTRACT_WEIGHT * abstract_hits) / len(terms)
    return coverage


def rank_papers(
    papers: list[Paper],
    queries: list[str],
    limit: int,
) -> list[Paper]:
    """Re-rank merged search results and keep the top `limit` papers."""
    if not papers:
        return []
    terms = plan_terms(queries)

    scored = [(score_paper(paper, terms), paper) for paper in papers]
    if scored:
        oldest = min(int(paper.published[:4]) for _, paper in scored if paper.published)
        rescored = []
        for score, paper in scored:
            year = int(paper.published[:4]) if paper.published else oldest
            bonus = min(RECENCY_CAP, RECENCY_BONUS * (year - oldest))
            rescored.append((score + bonus, paper))
    else:
        rescored = scored

    rescored.sort(key=lambda pair: pair[0], reverse=True)
    for score, paper in rescored[:limit]:
        print(f"  [ranked {score:.3f}] {paper.paper_id}  {paper.title[:55]}")
    return [paper for _, paper in rescored[:limit]]
