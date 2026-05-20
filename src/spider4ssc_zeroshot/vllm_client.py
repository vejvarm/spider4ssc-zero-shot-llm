from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, TypeVar

from huggingface_hub import model_info
from openai import APIConnectionError, APITimeoutError, OpenAI

T = TypeVar("T")


def _decoding_value(decoding: Any, key: str) -> Any:
    if isinstance(decoding, dict):
        return decoding[key]
    return getattr(decoding, key)


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
        self._model_revisions: dict[str, str] = {}
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
                if expected_model_id in model_ids:
                    return
                if model_ids:
                    last_error = RuntimeError(
                        f"vLLM endpoint served {model_ids}, expected {expected_model_id}"
                    )
            except (APIConnectionError, APITimeoutError) as exc:
                last_error = exc

            time.sleep(self.config.retry_sleep_seconds)

        message = f"Timed out waiting for vLLM endpoint at {self.config.base_url}"
        if last_error is not None:
            message = f"{message}: {last_error}"
            raise TimeoutError(message) from last_error
        raise TimeoutError(message)

    def complete(self, prompt: str, model_id: str, decoding: Any) -> dict[str, Any]:
        completion = self._with_endpoint_retries(
            lambda: self._client.chat.completions.create(
                model=model_id,
                messages=[{"role": "user", "content": prompt}],
                temperature=_decoding_value(decoding, "temperature"),
                top_p=_decoding_value(decoding, "top_p"),
                max_tokens=_decoding_value(decoding, "max_completion_tokens"),
                stop=_decoding_value(decoding, "stop"),
            )
        )
        choice = completion.choices[0]
        usage = completion.usage

        return {
            "raw_completion": choice.message.content or "",
            "finish_reason": choice.finish_reason,
            "usage": {
                "prompt_tokens": usage.prompt_tokens if usage else 0,
                "completion_tokens": usage.completion_tokens if usage else 0,
                "total_tokens": usage.total_tokens if usage else 0,
            },
            "model_revision": self._resolve_model_revision(model_id),
        }

    def _resolve_model_revision(self, model_id: str) -> str:
        if model_id not in self._model_revisions:
            self._model_revisions[model_id] = resolve_model_revision(model_id)
        return self._model_revisions[model_id]

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
