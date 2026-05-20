from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, TypeVar

from huggingface_hub import model_info
from openai import APIConnectionError, APITimeoutError, OpenAI

T = TypeVar("T")


@dataclass(frozen=True)
class VllmClientConfig:
    base_url: str = "http://localhost:8000/v1"
    api_key: str = "EMPTY"
    readiness_timeout_seconds: int = 1800
    request_timeout_seconds: int = 180
    max_retries: int = 5
    retry_sleep_seconds: int = 5


class VllmClient:
    def __init__(self, config: VllmClientConfig) -> None:
        self.config = config
        self._client = OpenAI(
            base_url=config.base_url,
            api_key=config.api_key,
            timeout=config.request_timeout_seconds,
        )

    def wait_until_ready(self, expected_model_id: str) -> None:
        deadline = time.monotonic() + self.config.readiness_timeout_seconds
        last_error: Exception | None = None

        while time.monotonic() < deadline:
            try:
                models = self._with_endpoint_retries(lambda: self._client.models.list())
                model_ids = [model.id for model in models.data]
                if expected_model_id in model_ids or model_ids:
                    return
            except (APIConnectionError, APITimeoutError) as exc:
                last_error = exc

            time.sleep(self.config.retry_sleep_seconds)

        message = f"Timed out waiting for vLLM endpoint at {self.config.base_url}"
        if last_error is not None:
            raise TimeoutError(message) from last_error
        raise TimeoutError(message)

    def complete(self, prompt: str, model_id: str, decoding: Any) -> dict[str, Any]:
        completion = self._with_endpoint_retries(
            lambda: self._client.chat.completions.create(
                model=model_id,
                messages=[{"role": "user", "content": prompt}],
                temperature=decoding.temperature,
                top_p=decoding.top_p,
                max_tokens=decoding.max_completion_tokens,
                stop=decoding.stop,
            )
        )
        choice = completion.choices[0]
        usage = completion.usage

        return {
            "raw_completion": choice.message.content or "",
            "finish_reason": choice.finish_reason,
            "usage": {
                "prompt_tokens": usage.prompt_tokens if usage else None,
                "completion_tokens": usage.completion_tokens if usage else None,
                "total_tokens": usage.total_tokens if usage else None,
            },
            "model_revision": resolve_model_revision(model_id),
        }

    def _with_endpoint_retries(self, operation: Callable[[], T]) -> T:
        attempts = self.config.max_retries + 1
        for attempt in range(attempts):
            try:
                return operation()
            except (APIConnectionError, APITimeoutError):
                if attempt == attempts - 1:
                    raise
                time.sleep(self.config.retry_sleep_seconds)

        raise RuntimeError("unreachable retry state")


def resolve_model_revision(model_id: str) -> str:
    return model_info(model_id).sha or "unknown"
