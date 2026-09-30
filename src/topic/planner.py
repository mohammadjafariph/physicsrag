"""
Stage 2 — Topic Planner.

Turns a user topic into a structured TopicPlan using any configured
LLM provider (see src/llm):

    Topic ("Measurement-induced phase transitions")
        -> Groq LLM (forced JSON output)
        -> TopicPlan {subtopics, search_queries}

The planner asks only one question: "What should we search for?"
(analysis of retrieved knowledge is the Research Analyzer's job, Stage 10).
"""

import json
import uuid
from dataclasses import dataclass

from config import settings
from src.llm.factory import build_llm


@dataclass
class Topic:
    """A research topic in the research tree.

    parent points to the topic this one was derived from (None for the
    root topic); cycle is the research iteration that produced it.
    """

    name: str
    parent: str | None = None
    cycle: int = 0
    topic_id: str = ""

    def __post_init__(self):
        if not self.topic_id:
            self.topic_id = str(uuid.uuid4())


@dataclass
class TopicPlan:
    """The structured output of the Topic Planner for one Topic.

    subtopics are the conceptual pieces of the topic;
    search_queries are the literal strings we send to arXiv (Stage 3).
    """

    topic: str
    subtopics: list[str]
    search_queries: list[str]


PLANNER_SYSTEM_PROMPT = (
    "You are a physics research planner. Given a physics research topic, "
    "you decompose it into focused subtopics and concrete arXiv search queries.\n\n"
    "Rules:\n"
    "- Stay strictly within physics research topics.\n"
    "- Subtopics: 5 to 8 short, specific research concepts (not general keywords).\n"
    "- Search queries: 3 to 5 queries written the way researchers search "
    "literature (technical phrases, no filler words).\n"
    "- Do not invent experiments, results, or papers. You only plan searches.\n\n"
    'Respond with JSON only, exactly this shape:\n'
    '{"subtopics": ["...", "..."], "search_queries": ["...", "..."]}'
)


def build_planner_prompt(topic_name: str) -> str:
    """Build the user message for the planner LLM."""
    return (
        f"Physics research topic: {topic_name}\n\n"
        "Decompose it into subtopics and arXiv search queries as JSON."
    )


def parse_plan_response(raw_text: str, topic_name: str) -> TopicPlan:
    """Parse the LLM's raw text into a TopicPlan.

    Raises ValueError with a readable message if the JSON is unusable,
    so callers fail loudly instead of passing empty plans downstream.
    """
    # Some models wrap JSON in markdown code fences even when told not to.
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("```")[1]
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as error:
        raise ValueError(f"Planner returned invalid JSON: {error}") from error

    subtopics = data.get("subtopics")
    search_queries = data.get("search_queries")

    if not isinstance(subtopics, list) or not isinstance(search_queries, list):
        raise ValueError(
            "Planner JSON missing 'subtopics' or 'search_queries' lists: "
            f"got keys {list(data.keys())}"
        )
    if not subtopics or not search_queries:
        raise ValueError("Planner returned empty subtopics or search_queries.")

    # Coerce everything to plain strings so downstream stages can rely on it.
    subtopics = [str(item).strip() for item in subtopics if str(item).strip()]
    search_queries = [str(item).strip() for item in search_queries if str(item).strip()]

    return TopicPlan(
        topic=topic_name,
        subtopics=subtopics,
        search_queries=search_queries,
    )


class TopicPlanner:
    """Calls the configured LLM provider and returns a validated TopicPlan."""

    def __init__(self, llm=None, model: str | None = None):
        self.llm = llm or build_llm()
        self.model = model or settings.stage_model(settings.planner_model)

    def plan(self, topic: Topic) -> TopicPlan:
        """Run the planner for one topic."""
        raw_text = self.llm.complete(
            [
                {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
                {"role": "user", "content": build_planner_prompt(topic.name)},
            ],
            model=self.model,
            temperature=0.2,  # low temperature -> stable, focused plans
            json_mode=True,   # force JSON output where the provider supports it
        )
        return parse_plan_response(raw_text, topic.name)
