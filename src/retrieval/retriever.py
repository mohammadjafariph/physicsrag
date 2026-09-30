"""
Stage 9 — retrieval engine.

Hybrid retrieval:

    query
      ├─ vector search   (Chroma, semantic similarity)
      └─ keyword search  (BM25, exact terms arXiv papers love)
           ↓
      Reciprocal Rank Fusion (RRF)
           ↓
    top-k Evidence {chunk + scores + provenance metadata}

Why hybrid: physics vocabulary is precise — BM25 catches exact terms
("Lindblad", "Zeno") that embeddings may blur; vector search catches
paraphrases BM25 cannot. `where` accepts a Chroma metadata filter.
"""

import math
import re
from dataclasses import dataclass, field

from src.database.vector_db import VectorStore
from src.embeddings.embedder import Embedder

TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z\-]+")

BM25_K1 = 1.5
BM25_B = 0.75
RRF_K = 60  # standard RRF constant
VECTOR_CANDIDATES = 15
KEYWORD_CANDIDATES = 15


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens; good enough for BM25 on physics text."""
    return TOKEN_RE.findall(text.lower())


@dataclass
class Evidence:
    """One retrieved chunk with provenance and both scores."""

    chunk_id: str
    paper_id: str
    title: str
    section: str
    page: int
    text: str
    topic: str = ""
    equations: str = ""         # verbatim LaTeX from the .tex source (Stage 5b)
    vector_score: float = 0.0
    keyword_score: float = 0.0
    fused_score: float = 0.0
    rerank_score: float = 0.0   # set by the cross-encoder reranker (Stage 19)


class BM25Index:
    """Minimal BM25 (Okapi) over {chunk_id: token list} documents."""

    def __init__(self, docs: dict[str, list[str]]):
        self.doc_tokens = docs
        self.doc_count = len(docs)
        self.avg_len = (
            sum(len(tokens) for tokens in docs.values()) / max(self.doc_count, 1)
        )
        self.doc_freq: dict[str, int] = {}
        for tokens in docs.values():
            for token in set(tokens):
                self.doc_freq[token] = self.doc_freq.get(token, 0) + 1

    def _idf(self, token: str) -> float:
        df = self.doc_freq.get(token, 0)
        if df == 0:
            return 0.0
        return math.log((self.doc_count - df + 0.5) / (df + 0.5) + 1.0)

    def search(self, query: str, k: int) -> dict[str, float]:
        """Return {chunk_id: bm25_score} for the top-k documents."""
        query_tokens = tokenize(query)
        scores: dict[str, float] = {}
        for chunk_id, tokens in self.doc_tokens.items():
            if not tokens:
                continue
            length_norm = BM25_K1 * (1 - BM25_B + BM25_B * len(tokens) / self.avg_len)
            token_counts: dict[str, int] = {}
            for token in tokens:
                token_counts[token] = token_counts.get(token, 0) + 1
            score = 0.0
            for token in query_tokens:
                tf = token_counts.get(token, 0)
                if tf:
                    score += self._idf(token) * tf * (BM25_K1 + 1) / (tf + length_norm)
            if score:
                scores[chunk_id] = score
        top = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        return dict(top[:k])


class Retriever:
    """Hybrid retriever over the VectorStore, with optional reranking.

    retrieve() = vector search + BM25 -> RRF fusion -> (optional)
    cross-encoder rerank -> top-k Evidence. `where` accepts a Chroma
    metadata filter, e.g. {"topic": "MIPT"} or {"paper_id": {"$in": [...]}}.
    """

    def __init__(
        self,
        store: VectorStore,
        embedder: Embedder,
        reranker=None,
    ):
        self.store = store
        self.embedder = embedder
        self.reranker = reranker
        self._bm25 = None
        self._texts: dict[str, str] = {}

    def refresh(self) -> None:
        """Rebuild the keyword index after the vector store changed."""
        pairs = self.store.get_all_texts()
        self._texts = dict(pairs)
        self._bm25 = BM25Index(
            {chunk_id: tokenize(text) for chunk_id, text in pairs}
        )

    @property
    def bm25(self) -> BM25Index:
        if self._bm25 is None:
            self.refresh()
        return self._bm25

    def retrieve(
        self,
        query: str,
        k: int = 8,
        where: dict | None = None,
        rerank: bool = True,
    ) -> list[Evidence]:
        """Return the top-k Evidence for one query (reranked by default)."""
        if self.store.count() == 0:
            return []

        query_embedding = self.embedder.embed_texts([query])[0]
        vector_hits = self.store.query(query_embedding, k=VECTOR_CANDIDATES, where=where)

        keyword_hits = self.bm25.search(query, k=KEYWORD_CANDIDATES)

        # Reciprocal Rank Fusion over the two ranked lists.
        fused: dict[str, Evidence] = {}
        for rank, hit in enumerate(vector_hits):
            evidence = Evidence(
                chunk_id=hit["chunk_id"], paper_id=hit["paper_id"],
                title=hit["title"], section=hit["section"], page=hit["page"],
                text=hit["text"], topic=hit.get("topic", ""),
                equations=hit.get("equations", ""),
                vector_score=hit["score"],
            )
            fused[evidence.chunk_id] = evidence
            evidence.fused_score += 1.0 / (RRF_K + rank + 1)
        for rank, (chunk_id, score) in enumerate(keyword_hits.items()):
            evidence = fused.setdefault(chunk_id, Evidence(
                chunk_id=chunk_id, paper_id=chunk_id.split(":")[0],
                title="", section="", page=0,
                text=self._texts.get(chunk_id, ""),
            ))
            evidence.keyword_score = score
            evidence.fused_score += 1.0 / (RRF_K + rank + 1)

        results = sorted(fused.values(), key=lambda e: e.fused_score, reverse=True)

        if rerank and self.reranker is not None:
            return self.reranker.rerank(query, results, k=k)
        return results[:k]


if __name__ == "__main__":
    from src.database.vector_db import VectorStore
    from src.embeddings.embedder import Embedder

    retriever = Retriever(VectorStore(), Embedder())
    for query in [
        "quantum Zeno effect from repeated measurements",
        "finite size scaling of entanglement entropy",
    ]:
        hits = retriever.retrieve(query, k=3)
        print(f"\nquery: {query!r}")
        for hit in hits:
            print(f"  fused={hit.fused_score:.4f} vec={hit.vector_score:.3f} "
                  f"bm25={hit.keyword_score:.2f} [{hit.chunk_id}] "
                  f"{hit.title[:45]} | {hit.section[:25]}")
