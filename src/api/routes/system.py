"""Health, library stats, and runtime provider settings."""

from fastapi import APIRouter, HTTPException

from config import BASE_DIR, ALL_PROVIDERS, settings
from src.api.deps import get_chat_service
from src.api.schemas import SettingsUpdate
from src.llm.base import LLMError
from src.llm.factory import build_llm, reset_default_llm
from src.memory.memory import ResearchMemory

router = APIRouter(tags=["system"])


def _counts() -> dict[str, int]:
    memory = ResearchMemory()
    try:
        topics = len(memory.all_topics())
    finally:
        memory.close()
    service = get_chat_service()
    return {
        "papers": len(list(settings.metadata_dir.glob("*.json"))),
        "chunks": service.store.count(),
        "topics": topics,
    }


@router.get("/api/health")
def health() -> dict:
    counts = _counts()
    return {
        "status": "ok",
        **counts,
        "provider": settings.llm_provider,
        "model": settings.llm_model,
    }


@router.get("/api/settings")
def get_settings() -> dict:
    try:
        base_url = settings.resolve_base_url()
    except ValueError:
        base_url = ""
    return {
        "provider": settings.llm_provider,
        "model": settings.llm_model,
        "base_url": base_url,
        "dialect": settings.dialect,
        "api_key_set": bool(settings.resolve_api_key()),
        "providers": ALL_PROVIDERS,
        "embedding_model": settings.embedding_model,
        "reranker_model": settings.reranker_model,
        "planner_model": settings.stage_model(settings.planner_model),
        "analyzer_model": settings.stage_model(settings.analyzer_model),
        "dedup_model": settings.stage_model(settings.dedup_model),
        "retrieval_k": settings.retrieval_k,
    }


@router.post("/api/settings")
def update_settings(update: SettingsUpdate) -> dict:
    """Switch provider/model at runtime; optionally persist to .env."""
    if update.provider is not None:
        provider = update.provider.lower().strip()
        if provider not in ALL_PROVIDERS:
            raise HTTPException(
                status_code=422,
                detail=f"unknown provider '{update.provider}'. "
                f"Known: {', '.join(ALL_PROVIDERS)}",
            )
        settings.llm_provider = provider
    if update.model is not None:
        settings.llm_model = update.model.strip()
    if update.base_url is not None:
        settings.llm_base_url = update.base_url.strip()

    key_changed = False
    if update.api_key:
        settings.llm_api_key = update.api_key.strip()
        key_changed = True

    reset_default_llm()

    if update.persist:
        from dotenv import set_key

        env_path = BASE_DIR / ".env"
        entries = {
            "LLM_PROVIDER": settings.llm_provider,
            "LLM_MODEL": settings.llm_model,
            "LLM_BASE_URL": settings.llm_base_url,
        }
        if key_changed:
            entries["LLM_API_KEY"] = settings.llm_api_key
        for name, value in entries.items():
            set_key(str(env_path), name, value)

    try:
        build_llm()  # validate the configuration now, not mid-question
    except LLMError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error

    return get_settings()