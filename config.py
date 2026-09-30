"""
Central configuration for PhysicsRAG.

Every knob is driven by environment variables (optionally via a .env file),
so the same code runs against any LLM provider or data path without edits.
Modules should import `settings` (or the DATA_DIR / PAPERS_DIR /
METADATA_DIR aliases for backwards compatibility) instead of touching
os.environ directly.
"""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent

# Provider presets: name -> (default base_url, API dialect).
#
# "openai_compatible" covers every provider speaking the OpenAI chat
# completions dialect (Groq, OpenRouter, OpenAI, Together, DeepSeek,
# Mistral, xAI, and local servers like Ollama, LM Studio, vLLM).
# Anthropic speaks its own dialect and gets a dedicated adapter.
# "custom" is your own OpenAI-compatible endpoint (set LLM_BASE_URL).
PROVIDER_PRESETS: dict[str, tuple[str, str]] = {
    "groq": ("https://api.groq.com/openai/v1", "openai_compatible"),
    "openai": ("https://api.openai.com/v1", "openai_compatible"),
    "openrouter": ("https://openrouter.ai/api/v1", "openai_compatible"),
    "deepseek": ("https://api.deepseek.com/v1", "openai_compatible"),
    "mistral": ("https://api.mistral.ai/v1", "openai_compatible"),
    "together": ("https://api.together.xyz/v1", "openai_compatible"),
    "xai": ("https://api.x.ai/v1", "openai_compatible"),
    "ollama": ("http://localhost:11434/v1", "openai_compatible"),
    "lmstudio": ("http://localhost:1234/v1", "openai_compatible"),
    "vllm": ("http://localhost:8001/v1", "openai_compatible"),
    "anthropic": ("https://api.anthropic.com", "anthropic"),
    "custom": ("", "openai_compatible"),
}

OPENAI_COMPATIBLE_PROVIDERS = sorted(
    name
    for name, (_, dialect) in PROVIDER_PRESETS.items()
    if dialect == "openai_compatible"
)
ALL_PROVIDERS = sorted(PROVIDER_PRESETS)


class Settings(BaseSettings):
    """All runtime configuration, overridable via environment / .env."""

    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ---- LLM provider -----------------------------------------------------
    llm_provider: str = "groq"
    llm_model: str = "openai/gpt-oss-120b"
    llm_api_key: str = ""           # env: LLM_API_KEY
    llm_base_url: str = ""          # env: LLM_BASE_URL, overrides the preset
    llm_temperature: float = 0.2
    llm_timeout: float = 180.0

    # Backwards compatibility: the original pipeline used GROQ_KEY. It is
    # still honoured when LLM_PROVIDER=groq and LLM_API_KEY is empty.
    groq_key: str = ""

    # ---- Per-stage model overrides (empty -> llm_model) --------------------
    planner_model: str = ""
    analyzer_model: str = ""
    dedup_model: str = ""

    # ---- Local models ------------------------------------------------------
    embedding_model: str = "all-MiniLM-L6-v2"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # ---- Retrieval / research budgets --------------------------------------
    retrieval_k: int = 8
    results_per_query: int = 5
    papers_per_cycle: int = 3
    evidence_chunks: int = 8
    download_delay: float = 3.0

    # ---- HTTP API ----------------------------------------------------------
    api_host: str = "127.0.0.1"
    api_port: int = 8000

    data_dir: Path = BASE_DIR / "data"

    # ---- Derived helpers -----------------------------------------------------

    @property
    def dialect(self) -> str:
        return PROVIDER_PRESETS.get(
            self.llm_provider.lower(), ("", "openai_compatible")
        )[1]

    @property
    def preset_base_url(self) -> str:
        return PROVIDER_PRESETS.get(self.llm_provider.lower(), ("", ""))[0]

    def resolve_base_url(self) -> str:
        url = self.llm_base_url.strip() or self.preset_base_url
        if not url:
            raise ValueError(
                f"provider '{self.llm_provider}' needs an explicit LLM_BASE_URL"
            )
        return url.rstrip("/")

    def resolve_api_key(self) -> str:
        if self.llm_api_key.strip():
            return self.llm_api_key.strip()
        if self.llm_provider.lower() == "groq":
            return self.groq_key.strip()
        return ""

    def stage_model(self, override: str) -> str:
        """Model for one pipeline stage, falling back to llm_model."""
        return (override or "").strip() or self.llm_model

    @property
    def papers_dir(self) -> Path:
        return self.data_dir / "papers"

    @property
    def metadata_dir(self) -> Path:
        return self.data_dir / "metadata"

    @property
    def latex_dir(self) -> Path:
        return self.data_dir / "latex"

    @property
    def vector_store_dir(self) -> Path:
        return self.data_dir / "vector_store"


settings = Settings()

# Directory aliases + creation on import, same contract as the original
# config.py: the downloader never crashes on a missing folder.
DATA_DIR = settings.data_dir
PAPERS_DIR = settings.papers_dir
METADATA_DIR = settings.metadata_dir
LATEX_DIR = settings.latex_dir
for _dir in (PAPERS_DIR, METADATA_DIR, LATEX_DIR):
    _dir.mkdir(parents=True, exist_ok=True)