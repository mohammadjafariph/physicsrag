"""
Stages 14+16+17 — research cycle controller, topic tree, provenance.

run_cycle() wires every stage into one iteration of the research loop:

    Topic -> TopicPlan -> Papers -> downloads -> chunks -> embeddings
          -> VectorStore -> evidence -> Analysis -> dedup -> Next Topic

Provenance (Stage 17) is preserved end-to-end:
- chunks keep paper_id/section/page in the vector store,
- every Analysis finding cites verified chunk ids,
- every cycle is persisted in SQLite with plan, papers, and analysis JSON.

The research tree (Stage 16) is reconstructed from topics' parent links.
"""

import json
from dataclasses import asdict, dataclass, field

from src.analysis.analyzer import Analysis, ResearchAnalyzer
from src.database.vector_db import VectorStore
from src.documents.chunker import chunk_document
from src.documents.downloader import download_papers
from src.documents.loader import load_paper
from src.embeddings.embedder import Embedder
from src.memory.memory import ResearchMemory
from src.retrieval.retriever import Retriever
from src.retrieval.ranker import Reranker
from src.search.arxiv_search import search_papers
from src.search.rank import rank_papers
from src.documents.downloader import DownloadResult
from src.search.paper import Paper
from src.topic.deduplicator import TopicDeduplicator
from src.topic.planner import Topic, TopicPlanner

NOVELTY_RANK = {"high": 0, "medium": 1, "low": 2}


@dataclass
class CycleConfig:
    """Budgets for one research cycle (keeps cycles fast and cheap)."""

    results_per_query: int = 5     # arXiv hits per search query
    papers_per_cycle: int = 3      # papers downloaded per cycle
    evidence_chunks: int = 8       # chunks retrieved for the analyzer
    download_delay: float = 3.0    # arXiv rate limit


@dataclass
class CycleSummary:
    """What one cycle did — printed and persisted."""

    cycle: int
    topic: str
    n_queries: int
    n_papers_found: int
    n_papers_downloaded: int
    n_new_chunks: int
    analysis: Analysis
    next_topic: str | None
    rejected: dict[str, str] = field(default_factory=dict)


class ResearchController:
    """Owns all components and runs single research cycles."""

    def __init__(
        self,
        memory: ResearchMemory | None = None,
        config: CycleConfig | None = None,
    ):
        self.config = config or CycleConfig()
        self.memory = memory or ResearchMemory()
        self.embedder = Embedder()
        self.store = VectorStore()
        self.retriever = Retriever(self.store, self.embedder, reranker=Reranker())
        self.planner = TopicPlanner()
        self.analyzer = ResearchAnalyzer()
        self.deduplicator = TopicDeduplicator()

    def run_cycle(self, topic: Topic, cycle_index: int) -> Topic | None:
        """One full research cycle; returns the next Topic or None."""
        print(f"\n{'=' * 70}\nCYCLE {cycle_index}: {topic.name}\n{'=' * 70}")

        # 1. Plan the searches for this topic.
        plan = self.planner.plan(topic)
        print(f"[plan] {len(plan.subtopics)} subtopics, "
              f"{len(plan.search_queries)} queries")

        # 2. Search arXiv for papers.
        papers = search_papers(
            plan.search_queries,
            results_per_query=self.config.results_per_query,
        )
        print(f"[search] {len(papers)} unique papers found")

        # 3. Filter/rank against the full plan, keep the most relevant few
        #    (Stage 19: protects the top-N cut from relaxed-query noise).
        selected = rank_papers(
            papers,
            plan.search_queries,
            limit=self.config.papers_per_cycle,
        )

        # 4. Download (cached papers are skipped internally).
        results = download_papers(selected, self.config.download_delay)
        ready: list[tuple[Paper, DownloadResult]] = [
            (paper, result) for paper, result in zip(selected, results)
            if result.pdf_path is not None
        ]
        print(f"[download] {len(ready)}/{len(selected)} papers on disk")

        # 5. Chunk + embed + store ONLY papers not already indexed.
        known_ids = self.store.paper_ids()
        new_chunks = []
        for paper, result in ready:
            if paper.paper_id in known_ids:
                continue
            document = load_paper(paper.paper_id)
            new_chunks.extend(chunk_document(document))
        if new_chunks:
            embeddings = self.embedder.embed_chunks(new_chunks)
            self.store.add_chunks(new_chunks, embeddings, topic=topic.name)
            self.retriever.refresh()
        print(f"[embed] {len(new_chunks)} new chunks stored "
              f"(collection: {self.store.count()})")

        # 6. Retrieve evidence for this topic.
        evidence = self.retriever.retrieve(topic.name, k=self.config.evidence_chunks)
        print(f"[retrieve] {len(evidence)} evidence chunks")

        # 7. Analyze: findings + gaps + candidate topics.
        analysis = self.analyzer.analyze(topic.name, evidence)
        print(f"[analyze] {len(analysis.established_findings)} findings, "
              f"{len(analysis.knowledge_gaps)} gaps, "
              f"{len(analysis.candidate_topics)} candidates")

        # 8. Deduplication (Stage 13): cheap prefilter, then LLM judge.
        candidate_names = [c.topic for c in analysis.candidate_topics
                           if c.topic.lower() != topic.name.lower()]
        prefiltered = []
        rejected: dict[str, str] = {}
        for name in candidate_names:
            embedding = self.embedder.embed_texts([name])[0]
            duplicate = self.memory.find_similar_topic(name, embedding)
            if duplicate:
                rejected[name] = f"prefilter: ~{duplicate}"
            else:
                prefiltered.append(name)
        novel_names, duplicates = self.deduplicator.filter_candidates(
            prefiltered, self.memory.all_topic_names()
        )
        for name, duplicate_of in {**duplicates}.items():
            rejected[name] = f"LLM judge: ~{duplicate_of}"
        for name in candidate_names:
            if name not in novel_names and name not in rejected:
                rejected[name] = "filtered"

        # 9. Select the next topic: highest novelty wins, stable order.
        novel = [
            candidate for candidate in analysis.candidate_topics
            if candidate.topic in novel_names
        ]
        novel.sort(key=lambda c: (NOVELTY_RANK.get(c.novelty, 3),))
        next_topic = None
        if novel:
            chosen = novel[0]
            next_topic = Topic(
                name=chosen.topic,
                parent=topic.name,
                cycle=cycle_index + 1,
            )
            embedding = self.embedder.embed_texts([next_topic.name])[0]
            self.memory.add_topic(next_topic, embedding)
            print(f"[next] {next_topic.name} (novelty: {chosen.novelty})")
            print(f"       why: {chosen.rationale[:110]}")
        else:
            print("[next] no novel topic found - stopping branch")

        # 10. Persist the whole cycle (provenance: analysis carries
        #     verified chunk ids; papers carry their arXiv ids).
        self.memory.add_cycle(
            cycle=cycle_index,
            topic=topic,
            plan=plan,
            papers=[
                {"paper_id": p.paper_id, "title": p.title,
                 "status": r.status}
                for p, r in ready
            ],
            # asdict recurses into Finding/KnowledgeGap dataclasses, so
            # provenance (chunk ids per claim) stays queryable in SQLite.
            analysis=asdict(analysis),
            next_topic=next_topic,
        )

        summary = CycleSummary(
            cycle=cycle_index,
            topic=topic.name,
            n_queries=len(plan.search_queries),
            n_papers_found=len(papers),
            n_papers_downloaded=len(ready),
            n_new_chunks=len(new_chunks),
            analysis=analysis,
            next_topic=next_topic.name if next_topic else None,
            rejected=rejected,
        )
        _print_summary(summary)
        return next_topic


def _print_summary(summary: CycleSummary) -> None:
    print(f"\n--- cycle {summary.cycle} summary ---")
    print(f"topic: {summary.topic}")
    print(f"queries: {summary.n_queries} | papers found: {summary.n_papers_found} "
          f"| downloaded: {summary.n_papers_downloaded} | new chunks: "
          f"{summary.n_new_chunks}")
    print("key findings:")
    for finding in summary.analysis.established_findings[:3]:
        print(f"  - {finding.claim[:100]}")
        print(f"    sources: {finding.source_paper_ids or '[no verified source]'}")
    for name, why in summary.rejected.items():
        print(f"  rejected: {name!r} ({why})")


def render_tree(memory: ResearchMemory) -> str:
    """Reconstruct the research tree from parent links (Stage 16)."""
    topics = memory.all_topics()
    if not topics:
        return "(no topics stored)"

    by_parent: dict[str | None, list[Topic]] = {}
    for t in topics:
        by_parent.setdefault(t.parent, []).append(t)

    lines: list[str] = []

    def walk(node: Topic, prefix: str = "", branch: str = "") -> None:
        marker = f"[cycle {node.cycle}] "
        lines.append(f"{prefix}{branch}{marker}{node.name}")
        children = by_parent.get(node.name, [])
        for index, child in enumerate(children):
            last = index == len(children) - 1
            walk(child,
                 prefix + ("    " if last or not prefix else "|   "),
                 "`-- " if last else "|-- ")

    for root in by_parent.get(None, []):
        walk(root)
    return "\n".join(lines)
