"""
Stage 19b — cross-encoder reranker (the final box of the Stage 9 diagram).

    Vector hits + BM25 hits -> RRF fusion -> CANDIDATE chunks
        -> CrossEncoder(query, chunk_text) for every candidate
        -> top-k evidence

Why a reranker: vector search embeds the query and the chunk separately
(bi-encoder); BM25 compares bags of words. Neither READS the pair
together. A cross-encoder feeds (query, chunk) through one transformer
and scores actual relevance — much more precise, but too slow to run on
the whole corpus, hence only on the fused candidate pool.

Model: cross-encoder/ms-marco-MiniLM-L-6-v2 (small, CPU-friendly).
"""

from sentence_transformers import CrossEncoder

from config import settings

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # default; settings wins


class Reranker:
    """Lazy-loading cross-encoder that reorders fused retrieval candidates."""

    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or settings.reranker_model
        self._model: CrossEncoder | None = None

    @property
    def model(self) -> CrossEncoder:
        if self._model is None:
            print(f"[reranker] loading model '{self.model_name}'...")
            self._model = CrossEncoder(self.model_name)
        return self._model

    def rerank(self, query: str, evidence: list, k: int | None = None) -> list:
        """Reorder evidence by query relevance; keeps all provenance fields.

        Each evidence object gets a rerank_score attribute; the returned
        list is sorted by it (descending) and truncated to k.
        """
        if not evidence:
            return []
        pairs = [(query, item.text) for item in evidence]
        scores = self.model.predict(pairs)
        for item, score in zip(evidence, scores):
            item.rerank_score = float(score)
        reranked = sorted(evidence, key=lambda item: item.rerank_score, reverse=True)
        return reranked[:k] if k else reranked
