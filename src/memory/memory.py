"""
Stages 13+15 — persistent research memory (SQLite) with topic deduplication.

ResearchMemory is the system's long-term record:

    topics    (id, name, parent, cycle, name-embedding)
    cycles    (per-cycle plan, analysis, papers, chosen next topic)

Deduplication (Stage 13): a candidate topic is rejected if its name
embedding is cosine-similar to ANY previously explored topic, breaking
the MIPT -> quantum measurement -> MIPT loop. Embeddings use the same
Embedder as chunks, so topic names and paper text share one space.
"""

import json
import re
import sqlite3
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from config import DATA_DIR
from src.topic.planner import Topic

DB_PATH = DATA_DIR / "research_memory.db"

DUPLICATE_THRESHOLD = 0.90  # conservative: embeddings alone can't judge paraphrases


def _is_acronym_of(acronym: str, name: str) -> bool:
    """True if `acronym` (e.g. 'MIPT') spells the initials of `name`."""
    words = re.findall(r"[A-Za-z]+", name)
    initials = "".join(word[0] for word in words if word).lower()
    return acronym.lower() == initials


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class ResearchMemory:
    """SQLite-backed memory of topics, cycles, and their artifacts."""

    def __init__(self, db_path: Path = DB_PATH):
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self._create_schema()

    def _create_schema(self) -> None:
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS topics (
                topic_id     TEXT PRIMARY KEY,
                name         TEXT NOT NULL,
                parent_name  TEXT,
                cycle        INTEGER NOT NULL,
                embedding    BLOB,
                created_at   TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS cycles (
                cycle            INTEGER PRIMARY KEY,
                topic_id         TEXT NOT NULL,
                topic_name       TEXT NOT NULL,
                plan_json        TEXT,
                papers_json      TEXT,
                analysis_json    TEXT,
                next_topic_id    TEXT,
                next_topic_name  TEXT,
                created_at       TEXT NOT NULL
            );
            """
        )
        self._migrate_schema()
        self.conn.commit()

    def _migrate_schema(self) -> None:
        """Upgrade older databases in place.

        v1 stored the parent TOPIC NAME in a column misleadingly called
        parent_id; it is renamed to parent_name (the value never changes).
        """
        columns = {
            row["name"]
            for row in self.conn.execute("PRAGMA table_info(topics)").fetchall()
        }
        if "parent_id" in columns:
            self.conn.execute(
                "ALTER TABLE topics RENAME COLUMN parent_id TO parent_name"
            )

    # ---- topics ------------------------------------------------------------

    def add_topic(self, topic: Topic, name_embedding: np.ndarray | None = None) -> str:
        """Store a topic; returns its id. Ignores exact-name duplicates."""
        existing = self.conn.execute(
            "SELECT topic_id FROM topics WHERE name = ?", (topic.name,)
        ).fetchone()
        if existing:
            return existing["topic_id"]

        blob = name_embedding.tobytes() if name_embedding is not None else None
        self.conn.execute(
            "INSERT INTO topics (topic_id, name, parent_name, cycle, embedding, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (topic.topic_id, topic.name, topic.parent, topic.cycle, blob, _now()),
        )
        self.conn.commit()
        return topic.topic_id

    def all_topics(self) -> list[Topic]:
        """Every stored topic (embedding NOT attached)."""
        rows = self.conn.execute(
            "SELECT topic_id, name, parent_name, cycle FROM topics ORDER BY cycle"
        ).fetchall()
        return [
            Topic(
                name=row["name"],
                parent=row["parent_name"],
                cycle=row["cycle"],
                topic_id=row["topic_id"],
            )
            for row in rows
        ]

    def known_embeddings(self) -> dict[str, np.ndarray]:
        """{topic_name: embedding} for similarity-based dedup."""
        rows = self.conn.execute(
            "SELECT name, embedding FROM topics WHERE embedding IS NOT NULL"
        ).fetchall()
        result = {}
        for row in rows:
            array = np.frombuffer(row["embedding"], dtype=np.float32)
            if array.size:
                result[row["name"]] = array
        return result

    def find_similar_topic(
        self,
        candidate_name: str,
        candidate_embedding: np.ndarray,
        threshold: float = DUPLICATE_THRESHOLD,
    ) -> str | None:
        """Cheap string-level duplicate check against stored topics.

        Only catches EXACT rewordings (high embedding threshold) and
        acronyms (MIPT = Measurement-Induced Phase Transitions). Genuine
        paraphrase judgment is the LLM deduplicator's job (Stage 13),
        because MiniLM cannot reliably separate dupes (sim ~0.65) from
        genuinely new subtopics (sim ~0.55) at short-name scale.
        """
        normalized = candidate_name.lower().strip()
        for name in self.all_topic_names():
            name_lower = name.lower().strip()
            if name_lower == normalized:
                return name
            if len(normalized) <= 6 and candidate_name.strip().isupper() \
                    and _is_acronym_of(normalized, name):
                return name
            if len(normalized) >= 12 and len(name_lower) >= 12 and (
                normalized in name_lower or name_lower in normalized
            ):
                return name
        for name, embedding in self.known_embeddings().items():
            # Guard against embeddings from a differently-sized model.
            if embedding.shape != candidate_embedding.shape:
                continue
            similarity = float(
                candidate_embedding @ embedding
                / (np.linalg.norm(candidate_embedding) * np.linalg.norm(embedding) + 1e-9)
            )
            if similarity >= threshold:
                return name
        return None

    def all_topic_names(self) -> list[str]:
        return [topic.name for topic in self.all_topics()]

    # ---- cycles ------------------------------------------------------------

    def add_cycle(
        self,
        cycle: int,
        topic: Topic,
        plan,
        papers: list[dict],
        analysis,
        next_topic: Topic | None,
    ) -> None:
        """Persist everything one research cycle produced."""
        # Accept either a dataclass TopicPlan or an already-dict plan.
        if plan is not None and not isinstance(plan, dict):
            plan = asdict(plan)
        # Accept either a dataclass Analysis or an already-dict analysis.
        if analysis is not None and not isinstance(analysis, dict):
            analysis = asdict(analysis)
        self.conn.execute(
            "INSERT OR REPLACE INTO cycles "
            "(cycle, topic_id, topic_name, plan_json, papers_json, analysis_json, "
            " next_topic_id, next_topic_name, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                cycle,
                topic.topic_id,
                topic.name,
                json.dumps(plan, ensure_ascii=False) if plan else None,
                json.dumps(papers, ensure_ascii=False),
                json.dumps(analysis, ensure_ascii=False) if analysis else None,
                next_topic.topic_id if next_topic else None,
                next_topic.name if next_topic else None,
                _now(),
            ),
        )
        self.conn.commit()

    def get_cycle(self, cycle: int) -> dict | None:
        row = self.conn.execute(
            "SELECT * FROM cycles WHERE cycle = ?", (cycle,)
        ).fetchone()
        return dict(row) if row else None

    def close(self) -> None:
        self.conn.close()


if __name__ == "__main__":
    import os
    from src.embeddings.embedder import Embedder

    # Use a throwaway DB for the demo.
    memory = ResearchMemory(DATA_DIR / "test_memory.db")
    embedder = Embedder()

    root = Topic(name="Measurement-induced phase transitions", cycle=0)
    memory.add_topic(root, embedder.embed_texts([root.name])[0])

    candidates = [
        "Measurement induced entanglement transitions",   # same topic (dupe)
        "Quantum trajectories",                            # genuinely new
        "MIPT",                                            # acronym dupe
    ]
    for candidate in candidates:
        duplicate = memory.find_similar_topic(
            candidate, embedder.embed_texts([candidate])[0]
        )
        verdict = f"DUPLICATE of '{duplicate}'" if duplicate else "NOVEL"
        print(f"{candidate!r:55} -> {verdict}")

    memory.close()
    os.remove(DATA_DIR / "test_memory.db")
