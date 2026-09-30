"""
Stage 7 — embedding generation.

Wraps sentence-transformers behind a small class:

    list[str] or list[Chunk]
        -> np.ndarray of float32 embeddings (one row per input)

Model choice: all-MiniLM-L6-v2
- 384 dimensions, fast on CPU, strong retrieval quality for its size.
- The same embedder is reused for topic similarity (Stage 13) so topics
  and chunks live in one vector space.
"""

import numpy as np
from sentence_transformers import SentenceTransformer

from config import settings
from src.documents.chunker import Chunk

MODEL_NAME = "all-MiniLM-L6-v2"  # default; settings.embedding_model wins


class Embedder:
    """Lazy-loading wrapper around one sentence-transformers model."""

    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or settings.embedding_model
        self._model: SentenceTransformer | None = None

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            print(f"[embedder] loading model '{self.model_name}'...")
            self._model = SentenceTransformer(self.model_name)
        return self._model

    @property
    def dimension(self) -> int:
        getter = getattr(self.model, "get_embedding_dimension", None)
        if getter is None:  # older sentence-transformers versions
            getter = self.model.get_sentence_embedding_dimension
        return getter()

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """Embed raw strings; returns (n, dim) float32, L2-normalized."""
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)
        return self.model.encode(
            texts,
            batch_size=32,
            normalize_embeddings=True,  # cosine similarity == dot product
            show_progress_bar=False,
        ).astype(np.float32)

    def embed_chunks(self, chunks: list[Chunk]) -> np.ndarray:
        """Embed chunks with a short context header for better retrieval."""
        texts = [
            f"{chunk.title}. Section: {chunk.section}. {chunk.text}"
            for chunk in chunks
        ]
        return self.embed_texts(texts)

    def similarity(self, text_a: str, text_b: str) -> float:
        """Cosine similarity between two strings (convenience helper)."""
        embeddings = self.embed_texts([text_a, text_b])
        return float(embeddings[0] @ embeddings[1])


if __name__ == "__main__":
    from src.documents.chunker import chunk_all_documents
    from src.documents.loader import load_available_papers

    chunks = chunk_all_documents(load_available_papers())
    embedder = Embedder()
    print("dimension:", embedder.dimension)

    embeddings = embedder.embed_chunks(chunks[:5])
    print("embeddings shape:", embeddings.shape)

    score = embedder.similarity(
        "measurement-induced phase transition",
        "entanglement transition in monitored quantum circuits",
    )
    print(f"similarity MIPT vs entanglement-transition phrasing: {score:.3f}")
