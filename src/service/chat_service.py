"""
The RAG answer pipeline behind the /api/chat endpoints.

    question -> hybrid retrieval (vector + BM25 -> RRF -> cross-encoder)
             -> numbered evidence block -> LLM answer with [n] citations
             -> citations parsed and mapped back to real chunk/paper ids

The LLM is provider-agnostic (src.llm): whichever provider/model is
configured answers. Reliability rule carried over from the analyzer: the
UI only ever shows citations that map to chunks actually retrieved.
"""

import re

from config import settings
from src.database.vector_db import VectorStore
from src.embeddings.embedder import Embedder
from src.llm.base import LLMError
from src.llm.factory import build_llm
from src.retrieval.retriever import Evidence, Retriever
from src.retrieval.ranker import Reranker

CITATION_RE = re.compile(r"\[(\d{1,2})\]")

CHAT_SYSTEM_PROMPT = """You are a physics research assistant answering questions from a private library of arXiv papers.

Rules:
- Answer ONLY from the numbered evidence excerpts you are given.
- Cite every claim with the excerpt number in square brackets, e.g. [2] or [1][3], placed directly after the claim it supports.
- If the evidence is insufficient for the question, say so plainly and state what is missing.
- Use LaTeX-style notation ($...$) for math.
- Be concise and precise; prefer equations and concrete results over vague prose.
"""


def arxiv_url_for(paper_id: str) -> str:
    return f"https://arxiv.org/abs/{paper_id}"


def build_evidence_block(evidence: list[Evidence]) -> str:
    """Format the retrieved chunks as a numbered, citation-ready block."""
    parts: list[str] = []
    for number, item in enumerate(evidence, start=1):
        label = (
            f"[{number}] chunk_id={item.chunk_id} | paper={item.paper_id} | "
            f"{item.title} | section: {item.section} | page {item.page}"
        )
        parts.append(f"{label}\n{item.text}")
    return "\n\n---\n\n".join(parts)


def source_dict(item: Evidence, number: int) -> dict:
    return {
        "n": number,
        "chunk_id": item.chunk_id,
        "paper_id": item.paper_id,
        "title": item.title,
        "section": item.section,
        "page": item.page,
        "arxiv_url": arxiv_url_for(item.paper_id),
    }


def parse_citations(answer_text: str, evidence: list[Evidence]) -> list[dict]:
    """[n] markers in the answer -> source dicts (first appearance order)."""
    cited: list[int] = []
    for match in CITATION_RE.finditer(answer_text):
        number = int(match.group(1))
        if 1 <= number <= len(evidence) and number not in cited:
            cited.append(number)
    return [source_dict(evidence[number - 1], number) for number in cited]


class ChatService:
    """Owns the retrieval stack; answers questions with citations."""

    def __init__(self):
        self.store = VectorStore()
        self.embedder = Embedder()
        self.retriever = Retriever(self.store, self.embedder, reranker=Reranker())

    def refresh(self) -> None:
        """Rebuild indexes after a research run added new papers."""
        self.retriever.refresh()

    def retrieve(self, question: str, k: int | None = None) -> list[Evidence]:
        return self.retriever.retrieve(question, k=k or settings.retrieval_k)

    @staticmethod
    def _messages(question: str, evidence: list[Evidence]) -> list[dict]:
        return [
            {"role": "system", "content": CHAT_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Question: {question}\n\n"
                    f"EVIDENCE ({len(evidence)} excerpts from the local paper "
                    f"library):\n\n{build_evidence_block(evidence)}"
                ),
            },
        ]

    def answer(self, question: str, k: int | None = None) -> dict:
        """One complete RAG answer (non-streaming)."""
        evidence = self.retrieve(question, k)
        if not evidence:
            raise ValueError(
                "no evidence retrieved — the paper library is empty; run a "
                "research cycle first"
            )
        llm = build_llm()
        answer_text = llm.complete(
            self._messages(question, evidence),
            temperature=settings.llm_temperature,
        )
        return {
            "question": question,
            "answer": answer_text,
            "citations": parse_citations(answer_text, evidence),
            "evidence_count": len(evidence),
            "provider": settings.llm_provider,
            "model": settings.llm_model,
        }

    def answer_stream(self, question: str, k: int | None = None) -> dict:
        """Yield SSE-ready events: citations -> tokens -> done/error."""
        evidence = self.retrieve(question, k)
        if not evidence:
            yield {
                "type": "error",
                "error": "no evidence retrieved — the paper library is "
                "empty; run a research cycle first",
            }
            return

        yield {
            "type": "citations",
            "sources": [source_dict(item, n) for n, item in enumerate(evidence, 1)],
        }

        collected: list[str] = []
        try:
            llm = build_llm()
            for token in llm.stream(
                self._messages(question, evidence),
                temperature=settings.llm_temperature,
            ):
                collected.append(token)
                yield {"type": "token", "text": token}
        except LLMError as error:
            yield {"type": "error", "error": str(error)}
            return

        yield {
            "type": "done",
            "citations": parse_citations("".join(collected), evidence),
        }