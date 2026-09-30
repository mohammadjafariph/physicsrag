"""Topic tree + cycle history endpoints (from research memory)."""

import json
from dataclasses import asdict

from fastapi import APIRouter

from src.controller.cycle import render_tree
from src.database.vector_db import VectorStore
from src.memory.memory import ResearchMemory

router = APIRouter(prefix="/api/topics", tags=["topics"])


@router.get("")
def list_topics() -> dict:
    memory = ResearchMemory()
    try:
        topics = [asdict(topic) for topic in memory.all_topics()]
        cycles = [
            dict(row)
            for row in memory.conn.execute(
                "SELECT cycle, topic_name, next_topic_name, created_at "
                "FROM cycles ORDER BY cycle"
            ).fetchall()
        ]
    finally:
        memory.close()
    return {"topics": topics, "cycles": cycles}


def _papers_by_topic(memory: ResearchMemory) -> dict[str, list[dict]]:
    """{topic_name: [paper, ...]} — papers as recorded per research cycle.

    A paper is listed under the topic of the FIRST cycle that recorded it
    (deduped globally by arXiv id), so every paper appears exactly once in
    the tree. chunk_count comes from the vector store (0 = not indexed).
    """
    by_topic: dict[str, list[dict]] = {}
    seen: set[str] = set()
    chunk_counts = VectorStore().paper_chunk_counts()
    rows = memory.conn.execute(
        "SELECT cycle, topic_name, papers_json FROM cycles ORDER BY cycle"
    ).fetchall()
    for cycle, topic_name, papers_json in rows:
        try:
            papers = json.loads(papers_json or "[]")
        except json.JSONDecodeError:
            continue
        for paper in papers:
            paper_id = paper.get("paper_id")
            if not paper_id or paper_id in seen:
                continue
            seen.add(paper_id)
            by_topic.setdefault(topic_name, []).append({
                "paper_id": paper_id,
                "title": paper.get("title") or paper_id,
                "status": paper.get("status") or "unknown",
                "cycle": cycle,
                "chunk_count": chunk_counts.get(paper_id, 0),
            })
    return by_topic


def _structured_tree(topics: list, papers_by_topic: dict[str, list[dict]] | None = None) -> list[dict]:
    """Parent-linked topic list -> nested nodes (children in cycle order)."""
    papers_by_topic = papers_by_topic or {}
    nodes = {
        topic.topic_id: {
            "topic_id": topic.topic_id,
            "name": topic.name,
            "parent": topic.parent,
            "cycle": topic.cycle,
            "papers": papers_by_topic.get(topic.name, []),
            "children": [],
        }
        for topic in topics
    }
    by_name = {node["name"]: node for node in nodes.values()}
    roots: list[dict] = []
    for node in nodes.values():
        parent = by_name.get(node["parent"]) if node["parent"] else None
        if parent is node:
            parent = None
        (parent["children"] if parent else roots).append(node)
    return roots


@router.get("/tree")
def topic_tree() -> dict:
    memory = ResearchMemory()
    try:
        topics = memory.all_topics()
        tree_text = render_tree(memory)
        papers_by_topic = _papers_by_topic(memory)
    finally:
        memory.close()
    return {
        "tree": tree_text,
        "nodes": _structured_tree(topics, papers_by_topic),
    }