"""Memory + deduplication tests (throwaway SQLite DB, tiny fake embeddings)."""

import numpy as np

from src.memory.memory import ResearchMemory
from src.topic.planner import Topic


def make_memory(tmp_path):
    memory = ResearchMemory(tmp_path / "test.db")
    memory.add_topic(
        Topic(name="Measurement-induced phase transitions", cycle=0),
        np.zeros(2, dtype=np.float32),
    )
    return memory


ZERO = np.zeros(2, dtype=np.float32)


def test_acronym_detected_as_duplicate(tmp_path):
    memory = make_memory(tmp_path)
    assert memory.find_similar_topic("MIPT", ZERO.copy()) == \
        "Measurement-induced phase transitions"
    memory.close()


def test_exact_name_detected(tmp_path):
    memory = make_memory(tmp_path)
    assert memory.find_similar_topic(
        "measurement-induced phase transitions", ZERO.copy()
    ) == "Measurement-induced phase transitions"
    memory.close()


def test_substring_detected(tmp_path):
    memory = make_memory(tmp_path)
    assert memory.find_similar_topic(
        "measurement-induced phase transitions in circuits", ZERO.copy()
    ) is not None
    memory.close()


def test_high_embedding_similarity_detected(tmp_path):
    memory = make_memory(tmp_path)
    same = np.array([1.0, 0.0], dtype=np.float32)
    memory.add_topic(Topic(name="topic a", cycle=1), same)
    assert memory.find_similar_topic("topic b", same.copy()) == "topic a"
    memory.close()


def test_genuinely_new_topic_is_novel(tmp_path):
    memory = make_memory(tmp_path)
    assert memory.find_similar_topic("bohmian neural network solvers", ZERO.copy()) is None
    memory.close()


def test_add_cycle_roundtrip(tmp_path):
    memory = make_memory(tmp_path)
    topic = Topic(name="Measurement-induced phase transitions", cycle=0)
    memory.add_cycle(
        cycle=0, topic=topic, plan={"subtopics": ["a"], "search_queries": ["b"]},
        papers=[{"paper_id": "0000.00001", "title": "t", "status": "cached"}],
        analysis={"established_findings": [], "key_concepts": []},
        next_topic=None,
    )
    row = memory.get_cycle(0)
    assert row["topic_name"] == topic.name
    assert '"paper_id": "0000.00001"' in row["papers_json"]
    memory.close()