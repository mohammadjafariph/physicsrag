"""Process-wide singletons shared by API routes."""

from functools import lru_cache

from src.service.chat_service import ChatService
from src.service.research_service import RunManager


@lru_cache(maxsize=1)
def get_chat_service() -> ChatService:
    """The retrieval stack is expensive (models) — one instance per process."""
    return ChatService()


@lru_cache(maxsize=1)
def get_run_manager() -> RunManager:
    """Research runs refresh the chat service's keyword index when done."""
    return RunManager(on_complete=lambda run: get_chat_service().refresh())