"""
The provider-agnostic LLM contract.

Every stage of the pipeline (planner, analyzer, deduplicator, chat) talks
to an LLMClient — never to a vendor SDK directly. Adapters translate this
contract to each provider dialect:

- OpenAICompatibleClient: every endpoint implementing the OpenAI chat
  completions API (Groq, OpenRouter, OpenAI, Together, DeepSeek, Mistral,
  xAI, Ollama, LM Studio, vLLM, custom gateways, ...)
- AnthropicClient: the Anthropic Messages API (Claude)

Swapping providers is therefore a config change, never a code change.
"""

from abc import ABC, abstractmethod
from typing import Iterator

Message = dict  # {"role": "system"|"user"|"assistant", "content": str}


class LLMError(RuntimeError):
    """Raised when a provider call fails or returns an unusable response."""


class LLMClient(ABC):
    """Minimal chat interface implemented by every provider adapter."""

    @abstractmethod
    def complete(
        self,
        messages: list[Message],
        *,
        model: str | None = None,
        temperature: float | None = None,
        json_mode: bool = False,
    ) -> str:
        """Return the full assistant reply as a string."""

    @abstractmethod
    def stream(
        self,
        messages: list[Message],
        *,
        model: str | None = None,
        temperature: float | None = None,
    ) -> Iterator[str]:
        """Yield the assistant reply incrementally as text chunks."""

    # ---- helpers shared by adapters ---------------------------------------

    @staticmethod
    def _split_system(
        messages: list[Message],
    ) -> tuple[str | None, list[Message]]:
        """Pull system messages out (Anthropic takes them separately)."""
        system_lines: list[str] = []
        rest: list[Message] = []
        for message in messages:
            if message.get("role") == "system":
                system_lines.append(message["content"])
            else:
                rest.append(message)
        joined = "\n\n".join(system_lines)
        return (joined or None), rest


def extract_json_object(raw: str) -> str:
    """Strip the markdown code fences some models wrap JSON in."""
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("```")[1]
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
    return cleaned.strip()