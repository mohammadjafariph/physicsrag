"""
Stage 13 (semantic part) — LLM topic deduplication.

Embedding similarity alone cannot judge short topic names (a true
paraphrase scores ~0.65 while a genuinely new subtopic scores ~0.55).
So the final novelty judgment is one Groq call:

    candidates + previously explored topics
        -> {topic, is_duplicate, duplicate_of} verdicts

This breaks loops like MIPT -> quantum measurement -> measurement-induced
transition -> MIPT.
"""

import json

from config import settings
from src.llm.base import extract_json_object
from src.llm.factory import build_llm

DEDUP_MODEL = "openai/gpt-oss-120b"

DEDUP_SYSTEM_PROMPT = """You are a physics topic deduplication judge.

You get a list of candidate research topics and a list of previously
explored topics. For each candidate decide whether it is essentially the
SAME research topic as one of the previously explored topics (a rewording,
an acronym, a trivially renamed variant) or genuinely NEW.

A candidate is a DUPLICATE when a physicist would say "we already studied
this". Judgement guide: when the PHYSICS MECHANISM and the PHENOMENON are
the same, it is a duplicate - "entanglement transitions induced by
measurements" duplicates "measurement-induced phase transitions", because
the transition and mechanism coincide. It is NEW when it narrows into a
different specific question, system, observable, or regime - even if it
shares words with an explored topic (e.g. "measurement rate dependence of
the purification dynamics" is NEW relative to the general MIPT topic).

Return JSON only:
{"verdicts": [{"topic": "<candidate>", "is_duplicate": true|false,
               "duplicate_of": "<explored topic or null>"}]}
One verdict per candidate, same order."""


class TopicDeduplicator:
    """LLM judge that filters duplicate candidate topics."""

    def __init__(self, llm=None, model: str | None = None):
        self.llm = llm or build_llm()
        self.model = model or settings.stage_model(settings.dedup_model)

    def filter_candidates(
        self,
        candidate_names: list[str],
        known_names: list[str],
    ) -> tuple[list[str], dict[str, str]]:
        """Return (novel_candidates, {candidate: duplicate_of}).

        Everything not judged a duplicate is kept, in input order.
        """
        if not candidate_names:
            return [], {}
        if not known_names:
            return list(candidate_names), {}

        user_message = (
            "Candidate topics:\n"
            + "\n".join(f"- {name}" for name in candidate_names)
            + "\n\nPreviously explored topics:\n"
            + "\n".join(f"- {name}" for name in known_names)
        )
        raw = self.llm.complete(
            [
                {"role": "system", "content": DEDUP_SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            model=self.model,
            temperature=0.0,
            json_mode=True,
        )

        data = json.loads(extract_json_object(raw))
        verdicts = data.get("verdicts", [])

        duplicates: dict[str, str] = {}
        for verdict in verdicts:
            if verdict.get("is_duplicate"):
                duplicates[str(verdict.get("topic", ""))] = str(
                    verdict.get("duplicate_of") or "unknown"
                )

        novel = [
            name for name in candidate_names
            if name not in duplicates
        ]
        return novel, duplicates


if __name__ == "__main__":
    deduplicator = TopicDeduplicator()
    known = ["Measurement-induced phase transitions"]
    candidates = [
        "Measurement induced entanglement transitions",   # rewording -> dupe
        "MIPT",                                            # acronym -> dupe
        "Higher-dimensional measurement-induced criticality",  # new angle
        "Quantum trajectories",                            # new
    ]
    novel, duplicates = deduplicator.filter_candidates(candidates, known)
    for candidate, duplicate_of in duplicates.items():
        print(f"DUPLICATE: {candidate!r} == {duplicate_of!r}")
    print("novel:", novel)
