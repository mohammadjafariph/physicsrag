"""Adapter for the Anthropic Messages API (Claude models)."""

from typing import Iterator

import anthropic

from src.llm.base import LLMClient, LLMError

JSON_SUFFIX = (
    "\n\nIMPORTANT: Respond with a single valid JSON object and nothing "
    "else. No markdown fences, no prose before or after."
)


class AnthropicClient(LLMClient):
    """Claude via the native Messages API.

    Anthropic has no server-side JSON mode, so json_mode is enforced by
    prompt instruction; the stage parsers (planner/analyzer/dedup) already
    fail loudly on unusable JSON.
    """

    def __init__(
        self,
        api_key: str,
        base_url: str | None = None,
        default_model: str = "",
        timeout: float = 180.0,
        max_retries: int = 2,
    ):
        self.client = anthropic.Anthropic(
            api_key=api_key,
            base_url=base_url or None,
            timeout=timeout,
            max_retries=max_retries,
        )
        self.default_model = default_model

    def _kwargs(
        self,
        messages: list[dict],
        model: str | None,
        temperature: float | None,
        json_mode: bool = False,
    ) -> dict:
        system, rest = self._split_system(messages)
        if json_mode:
            system = (system or "") + JSON_SUFFIX
        kwargs: dict = {
            "model": model or self.default_model,
            "max_tokens": 4096,
            "messages": rest,
        }
        if system:
            kwargs["system"] = system
        if temperature is not None:
            kwargs["temperature"] = temperature
        return kwargs

    def complete(
        self,
        messages: list[dict],
        *,
        model: str | None = None,
        temperature: float | None = None,
        json_mode: bool = False,
    ) -> str:
        kwargs = self._kwargs(messages, model, temperature, json_mode)
        try:
            response = self.client.messages.create(**kwargs)
        except anthropic.APIError as error:
            raise LLMError(f"provider request failed: {error}") from error
        text = "".join(
            block.text for block in response.content if block.type == "text"
        ).strip()
        if not text:
            raise LLMError("provider returned an empty response")
        return text

    def stream(
        self,
        messages: list[dict],
        *,
        model: str | None = None,
        temperature: float | None = None,
    ) -> Iterator[str]:
        kwargs = self._kwargs(messages, model, temperature)
        try:
            with self.client.messages.stream(**kwargs) as stream:
                yield from stream.text_stream
        except anthropic.APIError as error:
            raise LLMError(f"provider request failed: {error}") from error