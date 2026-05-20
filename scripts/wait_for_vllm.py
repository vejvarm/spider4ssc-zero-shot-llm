from __future__ import annotations

import argparse
import os
import time

from openai import APIConnectionError, APITimeoutError, OpenAI


def wait_until_ready(
    client: OpenAI,
    *,
    expected_model: str | None,
    timeout_seconds: int,
    sleep_seconds: int,
) -> list[str]:
    deadline = time.monotonic() + timeout_seconds
    last_error: Exception | None = None

    while time.monotonic() < deadline:
        try:
            models = client.models.list()
            model_ids = [model.id for model in models.data]
            if expected_model is None:
                if model_ids:
                    return model_ids
            elif expected_model in model_ids:
                return model_ids
            else:
                last_error = RuntimeError(
                    f"vLLM served models {model_ids}, expected {expected_model}"
                )
        except (APIConnectionError, APITimeoutError) as exc:
            last_error = exc
        time.sleep(sleep_seconds)

    message = f"vLLM did not become ready within {timeout_seconds} seconds"
    if expected_model is not None:
        message = f"{message}; expected {expected_model}"
    if last_error is not None:
        raise TimeoutError(message) from last_error
    raise TimeoutError(message)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--api-key", default=os.getenv("VLLM_API_KEY", "token-abc123"))
    parser.add_argument("--expected-model")
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    parser.add_argument("--sleep-seconds", type=int, default=5)
    args = parser.parse_args()

    client = OpenAI(base_url=args.base_url, api_key=args.api_key)
    print(
        wait_until_ready(
            client,
            expected_model=args.expected_model,
            timeout_seconds=args.timeout_seconds,
            sleep_seconds=args.sleep_seconds,
        )
    )


if __name__ == "__main__":
    main()
