"""
Stages 10-12 — Research Analyzer.

Given retrieved scientific evidence (Stage 9), the analyzer asks the two
questions the Topic Planner never asks:

    "What did we learn?"   -> established findings (WITH provenance)
    "What is missing?"     -> knowledge gaps + candidate next topics

Scientific reliability rules enforced here:
- Every finding MUST cite chunk_ids/paper_ids from the provided evidence.
- Candidate topics are stored as ResearchDirection objects with a
  "hypothesis" flavor — they are LLM proposals, never established facts.
- Contradictions between papers are preserved, not merged.
"""

import json
from dataclasses import dataclass, field

from config import settings
from src.llm.factory import build_llm
from src.retrieval.retriever import Evidence

ANALYZER_MODEL = "openai/gpt-oss-120b"

MAX_EVIDENCE_CHARS = 24_000  # context budget for the evidence block

ANALYZER_SYSTEM_PROMPT = """You are a research analyst for a physics literature exploration system.

You receive numbered evidence excerpts retrieved from real arXiv papers,
each labeled with its source id like [chunk_id | paper_id | Title | section].

Analyze the evidence for ONE research topic and return JSON only, exactly:

{
  "established_findings": [
    {"claim": "...",
     "source_chunk_ids": ["<chunk ids that appear in the evidence>"],
     "source_paper_ids": ["<arxiv ids>"]}
  ],
  "key_concepts": ["..."],
  "contradictions": ["... claims that conflict between papers, or 'none observed'"],
  "unresolved_questions": ["..."],
  "knowledge_gaps": [
    {"description": "...", "related_paper_ids": ["..."]}
  ],
  "candidate_topics": [
    {"topic": "short specific topic name (max 8 words)",
     "rationale": "why this is worth exploring next, grounded in the evidence",
     "novelty": "high|medium|low"}
  ],
  "evidence_summary": "2-4 sentence summary of what the evidence covers"
}

STRICT RULES:
- Every claim MUST come from the evidence. Cite only chunk ids you were given.
- If evidence does not support a finding, do not include it.
- Propose 3-5 candidate topics that probe knowledge gaps, weakly explored
  connections, or open experimental questions - not generic related topics.
- Do not invent papers, results, or citations."""


@dataclass
class Finding:
    """An established claim extracted from evidence, with provenance."""

    claim: str
    source_chunk_ids: list[str] = field(default_factory=list)
    source_paper_ids: list[str] = field(default_factory=list)


@dataclass
class KnowledgeGap:
    """A hole in the accumulated knowledge of the system."""

    description: str
    related_paper_ids: list[str] = field(default_factory=list)


@dataclass
class ResearchDirection:
    """An LLM-PROPOSED next topic (a hypothesis, not an established fact)."""

    topic: str
    rationale: str
    novelty: str = "medium"


@dataclass
class Analysis:
    """Structured result of analyzing retrieved evidence for one topic."""

    topic: str
    established_findings: list[Finding]
    key_concepts: list[str]
    contradictions: list[str]
    unresolved_questions: list[str]
    knowledge_gaps: list[KnowledgeGap]
    candidate_topics: list[ResearchDirection]
    evidence_summary: str
    analyzed_evidence_chunk_ids: list[str]


def build_evidence_block(evidence: list[Evidence]) -> str:
    """Format retrieved chunks with source labels, inside a char budget."""
    parts: list[str] = []
    used = 0
    for item in evidence:
        label = f"[{item.chunk_id} | {item.paper_id} | {item.title} | {item.section}]"
        part = f"{label}\n{item.text}\n"
        if used + len(part) > MAX_EVIDENCE_CHARS:
            break
        parts.append(part)
        used += len(part)
    return "\n---\n".join(parts)


def parse_analysis(raw: str, topic: str, evidence_chunk_ids: list[str]) -> Analysis:
    """Parse the LLM response into an Analysis; fail loudly on bad JSON."""
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("```")[1]
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
    data = json.loads(cleaned)

    known_chunk_ids = set(evidence_chunk_ids)
    findings = []
    for item in data.get("established_findings", []):
        claim = str(item.get("claim", "")).strip()
        if not claim:
            continue
        # Keep only chunk ids that actually exist in the evidence, then
        # DERIVE paper ids from them (chunk_id prefix = arXiv id). The
        # model's own paper attribution is not trusted: it can be wrong.
        chunk_ids = [
            c for c in item.get("source_chunk_ids", []) if c in known_chunk_ids
        ]
        paper_ids = sorted({chunk_id.split(":")[0] for chunk_id in chunk_ids})
        findings.append(
            Finding(
                claim=claim,
                source_chunk_ids=chunk_ids,
                source_paper_ids=paper_ids,
            )
        )
    gaps = [
        KnowledgeGap(
            description=str(item.get("description", "")).strip(),
            related_paper_ids=[str(p) for p in item.get("related_paper_ids", [])],
        )
        for item in data.get("knowledge_gaps", [])
        if str(item.get("description", "")).strip()
    ]
    candidates = [
        ResearchDirection(
            topic=str(item.get("topic", "")).strip(),
            rationale=str(item.get("rationale", "")).strip(),
            novelty=str(item.get("novelty", "medium")).strip().lower(),
        )
        for item in data.get("candidate_topics", [])
        if str(item.get("topic", "")).strip()
    ]

    return Analysis(
        topic=topic,
        established_findings=findings,
        key_concepts=[str(c) for c in data.get("key_concepts", [])],
        contradictions=[str(c) for c in data.get("contradictions", [])],
        unresolved_questions=[str(q) for q in data.get("unresolved_questions", [])],
        knowledge_gaps=gaps,
        candidate_topics=candidates,
        evidence_summary=str(data.get("evidence_summary", "")).strip(),
        analyzed_evidence_chunk_ids=evidence_chunk_ids,
    )


class ResearchAnalyzer:
    """Analyzes retrieved evidence into an Analysis via any configured provider."""

    def __init__(self, llm=None, model: str | None = None):
        self.llm = llm or build_llm()
        self.model = model or settings.stage_model(settings.analyzer_model)

    def analyze(self, topic: str, evidence: list[Evidence]) -> Analysis:
        evidence_block = build_evidence_block(evidence)
        if not evidence_block.strip():
            raise ValueError("analyze() called with empty evidence")

        user_message = (
            f"Research topic: {topic}\n\n"
            f"EVIDENCE ({len(evidence)} excerpts from real arXiv papers):\n\n"
            f"{evidence_block}"
        )

        raw = self.llm.complete(
            [
                {"role": "system", "content": ANALYZER_SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            model=self.model,
            temperature=0.2,
            json_mode=True,
        )
        chunk_ids = [item.chunk_id for item in evidence]
        return parse_analysis(raw, topic, chunk_ids)


if __name__ == "__main__":
    from src.database.vector_db import VectorStore
    from src.embeddings.embedder import Embedder
    from src.retrieval.retriever import Retriever

    retriever = Retriever(VectorStore(), Embedder())
    evidence = retriever.retrieve("measurement-induced phase transition criticality", k=8)

    analyzer = ResearchAnalyzer()
    analysis = analyzer.analyze("Measurement-induced phase transitions", evidence)

    print(f"evidence summary: {analysis.evidence_summary}\n")
    print("ESTABLISHED FINDINGS (with provenance):")
    for finding in analysis.established_findings:
        print(f"- {finding.claim}")
        print(f"  sources: {finding.source_paper_ids} chunks={finding.source_chunk_ids}")
    print(f"\nKEY GAPS ({len(analysis.knowledge_gaps)}):")
    for gap in analysis.knowledge_gaps:
        print(f"- {gap.description}")
    print(f"\nCANDIDATE NEXT TOPICS:")
    for candidate in analysis.candidate_topics:
        print(f"- {candidate.topic} (novelty: {candidate.novelty})")
        print(f"  {candidate.rationale[:120]}")
