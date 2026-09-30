"""Settings / provider-preset tests (no network)."""

from config import ALL_PROVIDERS, PROVIDER_PRESETS, Settings


def test_preset_resolution_groq():
    s = Settings(_env_file=None, llm_provider="groq", groq_key="test-key")
    assert s.resolve_base_url() == "https://api.groq.com/openai/v1"
    assert s.resolve_api_key() == "test-key"
    assert s.dialect == "openai_compatible"


def test_legacy_groq_key_alias():
    s = Settings(_env_file=None, llm_provider="groq", llm_api_key="", groq_key="legacy")
    assert s.resolve_api_key() == "legacy"


def test_base_url_override_strips_trailing_slash():
    s = Settings(_env_file=None, llm_provider="custom", llm_base_url="http://x:9999/v1/")
    assert s.resolve_base_url() == "http://x:9999/v1"


def test_unknown_provider_defaults_to_openai_compatible():
    s = Settings(_env_file=None, llm_provider="something-new")
    assert s.dialect == "openai_compatible"


def test_explicit_llm_api_key_wins():
    s = Settings(_env_file=None, llm_provider="openai", llm_api_key="main", groq_key="legacy")
    assert s.resolve_api_key() == "main"


def test_stage_model_fallback():
    s = Settings(_env_file=None, llm_model="main-model", planner_model="")
    assert s.stage_model(s.planner_model) == "main-model"
    assert s.stage_model("override") == "override"


def test_custom_without_base_url_raises():
    s = Settings(_env_file=None, llm_provider="custom", llm_base_url="")
    try:
        s.resolve_base_url()
    except ValueError as error:
        assert "LLM_BASE_URL" in str(error)
    else:
        raise AssertionError("expected ValueError")


def test_all_presets_have_dialects():
    assert len(ALL_PROVIDERS) == len(PROVIDER_PRESETS)
    for _, (_, dialect) in PROVIDER_PRESETS.items():
        assert dialect in {"openai_compatible", "anthropic"}