"""
Build the right LLMClient from Settings (or explicit overrides).

    build_llm()                          # from settings / .env
    build_llm(provider="ollama")         # runtime override (UI settings)

The same factory serves every pipeline stage, so switching provider or
model is a config change — never a code change.
"""

from functools import lru_cache

from config import PROVIDER_PRESETS, settings
from src.llm.anthropic_client import AnthropicClient
from src.llm.base import LLMClient, LLMError
from src.llm.openai_compat import OpenAICompatibleClient


def build_llm(
    provider: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
) -> LLMClient:
    """Construct a client; arguments override settings. Raises LLMError
    with an actionable message when the configuration is unusable."""
    provider_name = (provider or settings.llm_provider or "custom").lower().strip()

    if provider_name not in PROVIDER_PRESETS:
        known = ", ".join(sorted(PROVIDER_PRESETS))
        raise LLMError(f"unknown provider '{provider_name}'. Known providers: {known}")

    preset_url, dialect = PROVIDER_PRESETS[provider_name]
    resolved_url = (
        base_url or settings.llm_base_url or preset_url or ""
    ).strip().rstrip("/")
    resolved_model = (model or settings.llm_model or "").strip()
    if api_key:
        resolved_key = api_key.strip()
    else:
        resolved_key = settings.resolve_api_key()

    if not resolved_url:
        raise LLMError(
            f"no base URL for provider '{provider_name}': set LLM_BASE_URL"
        )
    if not resolved_model:
        raise LLMError("no model configured: set LLM_MODEL")

    if dialect == "anthropic":
        if not resolved_key:
            raise LLMError(
                "Anthropic provider needs an API key: set LLM_API_KEY"
            )
        return AnthropicClient(
            api_key=resolved_key,
            base_url=resolved_url,
            default_model=resolved_model,
            timeout=settings.llm_timeout,
        )

    return OpenAICompatibleClient(
        base_url=resolved_url,
        api_key=resolved_key,
        default_model=resolved_model,
        timeout=settings.llm_timeout,
    )


@lru_cache(maxsize=1)
def default_llm() -> LLMClient:
    """Cached default client; rebuilt after runtime settings change."""
    return build_llm()


def reset_default_llm() -> None:
    """Drop the cached client (called after runtime settings change)."""
    default_llm.cache_clear()