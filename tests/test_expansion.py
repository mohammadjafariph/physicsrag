"""Expansion helpers: root resolution and cycle numbering."""

import numpy as np

from src.memory.memory import ResearchMemory
from src.service.research_service import next_base_cycle, resolve_root_topic
from src.topic.planner import Topic


def make_memory(tmp_path):
    return ResearchMemory(tmp_path / "expansion.db")


def test_new_topic_gets_fresh_root(tmp_path):
    memory = make_memory(tmp_path)
    topic, expanded = resolve_root_topic(memory, "Some brand new topic")
    assert expanded is False
    assert topic.parent is None and topic.cycle == 0


def test_existing_topic_is_reused_with_its_branch(tmp_path):
    memory = make_memory(tmp_path)
    stored = Topic(name="MIPT open quantum systems", parent="MIPT", cycle=3)
    memory.add_topic(stored)
    topic, expanded = resolve_root_topic(memory, "MIPT open quantum systems")
    assert expanded is True
    assert topic.parent == "MIPT" and topic.cycle == 3
    assert topic.name == "MIPT open quantum systems"


def test_next_base_cycle_empty_db_starts_at_zero(tmp_path):
    memory = make_memory(tmp_path)
    assert next_base_cycle(memory) == 0


def test_next_base_cycle_continues_after_history(tmp_path):
    memory = make_memory(tmp_path)
    for cycle in (0, 1, 2):
        memory.add_cycle(
            cycle=cycle,
            topic=Topic(name=f"t{cycle}", cycle=cycle),
            plan=None, papers=[], analysis=None, next_topic=None,
        )
    assert next_base_cycle(memory) == 3
