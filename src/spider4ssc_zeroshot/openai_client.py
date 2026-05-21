from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, TypeVar

from openai import APIConnectionError, APITimeoutError, OpenAI

T = TypeVar("T")


def _decoding_value(decoding: Any, key: str, default: Any = None) -> Any:
    if isinstance(decoding, dict):
        return decoding.get(key, default)
    return getattr(decoding, key, default)


@dataclass(frozen=True)
class OpenAIChatClientConfig:
    base_url: str = "https://api.openai.com/v1"
    api_key: str = ""
    request_timeout_seconds: int = 180
    max_retries: int = 5
    retry_sleep_seconds: int = 5


class OpenAIChatClient:
    def __init__(self, config: OpenAIChatClientConfig) -> None:
        self.config = config
        self._client = OpenAI(
            base_url=config.base_url,
            api_key=config.api_key,
            timeout=config.request_timeout_seconds,
        )

    def wait_until_ready(self, expected_model_id: str) -> None:
        return None

    def complete(self, prompt: str, model_id: str, decoding: Any) -> dict[str, Any]:
        request = {
            "model": model_id,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": _decoding_value(decoding, "temperature"),
            "top_p": _decoding_value(decoding, "top_p"),
            "max_completion_tokens": _decoding_value(decoding, "max_completion_tokens"),
            "stop": _decoding_value(decoding, "stop"),
            "store": False,
        }
        reasoning_effort = _decoding_value(decoding, "reasoning_effort")
        if reasoning_effort is not None:
            request["reasoning_effort"] = reasoning_effort

        completion = self._with_endpoint_retries(
            lambda: self._client.chat.completions.create(**request)
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
            "model_revision": getattr(completion, "model", model_id),
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
