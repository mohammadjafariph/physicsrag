"""
Stage 8 — vector database (Chroma, persistent, local).

    add_chunks(chunks, embeddings)   -> chunks become searchable
    query(text_or_embedding, k, where) -> ranked chunks with metadata

Storage lives in data/vector_store/. We pass OUR OWN embeddings
(Stage 7 embedder) rather than letting Chroma embed with its default
model, so retrieval and topic-similarity share one vector space.
"""

from pathlib import Path

import chromadb

from config import DATA_DIR
from src.documents.chunker import Chunk

VECTOR_STORE_DIR = DATA_DIR / "vector_store"

COLLECTION_NAME = "chunks"


class VectorStore:
    """Thin wrapper around a persistent Chroma collection of paper chunks."""

    def __init__(
        self,
        persist_dir: Path = VECTOR_STORE_DIR,
        collection_name: str = COLLECTION_NAME,
    ):
        self.client = chromadb.PersistentClient(path=str(persist_dir))
        # Cosine space: embeddings are L2-normalized, so this matches
        # the dot-product similarities computed everywhere else.
        self.collection = self.client.get_or_create_collection(
            collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def count(self) -> int:
        return self.collection.count()

    @staticmethod
    def _chunk_metadata(chunk: Chunk, topic: str = "") -> dict:
        """Chroma metadata must be scalars; lists are string-joined."""
        return {
            "paper_id": chunk.paper_id,
            "title": chunk.title,
            "section": chunk.section,
            "page": chunk.page,
            "index": chunk.index,
            "topic": topic,
        }

    def add_chunks(
        self,
        chunks: list[Chunk],
        embeddings,
        topic: str = "",
    ) -> int:
        """Insert (or update) chunks with their embeddings. Returns count."""
        if not chunks:
            return 0
        self.collection.upsert(
            ids=[chunk.chunk_id for chunk in chunks],
            embeddings=[e.tolist() for e in embeddings],
            documents=[chunk.text for chunk in chunks],
            metadatas=[self._chunk_metadata(chunk, topic) for chunk in chunks],
        )
        return len(chunks)

    def query(
        self,
        query_embedding,
        k: int = 5,
        where: dict | None = None,
    ) -> list[dict]:
        """Nearest chunks to an embedding. Returns a list of result dicts:
        {chunk_id, text, score, paper_id, title, section, page, topic}."""
        if self.count() == 0:
            return []
        result = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=min(k, self.count()),
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        hits = []
        for chunk_id, document, metadata, distance in zip(
            result["ids"][0],
            result["documents"][0],
            result["metadatas"][0],
            result["distances"][0],
        ):
            hits.append({
                "chunk_id": chunk_id,
                "text": document,
                "score": 1.0 - float(distance),  # cosine distance -> similarity
                **metadata,
            })
        return hits

    def get_all_texts(self) -> list[tuple[str, str]]:
        """All (chunk_id, text) pairs — used by the keyword retriever."""
        if self.count() == 0:
            return []
        stored = self.collection.get(include=["documents"])
        return list(zip(stored["ids"], stored["documents"]))

    def paper_ids(self) -> set[str]:
        """ArXiv IDs already embedded — lets the controller skip re-work."""
        if self.count() == 0:
            return set()
        stored = self.collection.get(include=["metadatas"])
        return {m["paper_id"] for m in stored["metadatas"]}

    def paper_chunk_counts(self) -> dict[str, int]:
        """{paper_id: number of stored chunks} for UI listings."""
        if self.count() == 0:
            return {}
        stored = self.collection.get(include=["metadatas"])
        counts: dict[str, int] = {}
        for metadata in stored["metadatas"]:
            paper_id = metadata.get("paper_id", "")
            counts[paper_id] = counts.get(paper_id, 0) + 1
        return counts

    def get_paper_chunks(self, paper_id: str) -> list[dict]:
        """All stored chunks of one paper, ordered by index within it."""
        stored = self.collection.get(
            where={"paper_id": paper_id},
            include=["documents", "metadatas"],
        )
        chunks = [
            {"chunk_id": chunk_id, "text": document, **metadata}
            for chunk_id, document, metadata in zip(
                stored["ids"], stored["documents"], stored["metadatas"]
            )
        ]
        chunks.sort(key=lambda chunk: chunk.get("index", 0))
        return chunks


if __name__ == "__main__":
    from src.documents.chunker import chunk_all_documents
    from src.documents.loader import load_available_papers
    from src.embeddings.embedder import Embedder

    chunks = chunk_all_documents(load_available_papers())
    embedder = Embedder()
    store = VectorStore()

    embeddings = embedder.embed_chunks(chunks)
    added = store.add_chunks(chunks, embeddings, topic="MIPT")
    print(f"added {added} chunks, collection now holds {store.count()}")

    query = "how does the entanglement entropy scale at the critical point"
    query_embedding = embedder.embed_texts([query])[0]
    hits = store.query(query_embedding, k=3)
    print(f"\ntop 3 for: {query!r}")
    for hit in hits:
        print(f"  {hit['score']:.3f} [{hit['chunk_id']}] {hit['title'][:50]} "
              f"| {hit['section'][:30]}")
