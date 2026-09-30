"""Topic tree + cycle history endpoints (from research memory)."""

from dataclasses import asdict

from fastapi import APIRouter

from src.controller.cycle import render_tree
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


def _structured_tree(topics: list) -> list[dict]:
    """Parent-linked topic list -> nested nodes (children in cycle order)."""
    nodes = {
        topic.topic_id: {
            "topic_id": topic.topic_id,
            "name": topic.name,
            "parent": topic.parent,
            "cycle": topic.cycle,
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
    finally:
        memory.close()
    return {"tree": tree_text, "nodes": _structured_tree(topics)}