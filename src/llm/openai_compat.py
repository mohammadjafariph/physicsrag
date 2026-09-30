"""
Adapter for every OpenAI-compatible chat-completions endpoint.

One client class covers Groq, OpenAI, OpenRouter, Together, DeepSeek,
Mistral, xAI, and local servers (Ollama, LM Studio, vLLM): they all
implement the same /chat/completions dialect; only the base URL changes.
"""

import openai
from typing import Iterator

from src.llm.base import LLMClient, LLMError


class OpenAICompatibleClient(LLMClient):
    """OpenAI dialect client bound to one base URL / model."""

    def __init__(
        self,
        base_url: str,
        api_key: str = "",
        default_model: str = "",
        timeout: float = 180.0,
        max_retries: int = 2,
    ):
        self.client = openai.OpenAI(
            base_url=base_url,
            # Local servers (Ollama, LM Studio) ignore the key but the SDK
            # wants a non-empty string.
            api_key=api_key or "not-needed",
            timeout=timeout,
            max_retries=max_retries,
        )
        self.default_model = default_model

    @staticmethod
    def _payload(
        messages: list[dict],
        model: str | None,
        temperature: float | None,
        json_mode: bool,
    ) -> dict:
        payload: dict = {"messages": messages}
        if model:
            payload["model"] = model
        if temperature is not None:
            payload["temperature"] = temperature
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        return payload

    def complete(
        self,
        messages: list[dict],
        *,
        model: str | None = None,
        temperature: float | None = None,
        json_mode: bool = False,
    ) -> str:
        payload = self._payload(messages, model or self.default_model, temperature, json_mode)
        try:
            try:
                response = self.client.chat.completions.create(**payload)
            except openai.BadRequestError as error:
                # Not every compatible server supports response_format;
                # retry once without it (the caller's prompt already asks
                # for JSON, and stage parsers fail loudly on bad JSON).
                if json_mode and "response_format" in str(error):
                    payload.pop("response_format", None)
                    response = self.client.chat.completions.create(**payload)
                else:
                    raise
        except openai.APIError as error:
            raise LLMError(f"provider request failed: {error}") from error

        content = response.choices[0].message.content if response.choices else None
        if not content:
            raise LLMError("provider returned an empty response")
        return content

    def stream(
        self,
        messages: list[dict],
        *,
        model: str | None = None,
        temperature: float | None = None,
    ) -> Iterator[str]:  # type: ignore[valid-type]
        from typing import Iterator  # noqa: PLC0415

        payload = self._payload(messages, model or self.default_model, temperature, False)
        try:
            chunks = self.client.chat.completions.create(**payload, stream=True)
            for chunk in chunks:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta
        except openai.APIError as error:
            raise LLMError(f"provider request failed: {error}") from error