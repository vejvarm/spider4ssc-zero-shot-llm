from __future__ import annotations

import argparse
import os
import time

from openai import APIConnectionError, APITimeoutError, OpenAI


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--api-key", default=os.getenv("VLLM_API_KEY", "token-abc123"))
    parser.add_argument("--expected-model")
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    parser.add_argument("--sleep-seconds", type=int, default=5)
    args = parser.parse_args()

    client = OpenAI(base_url=args.base_url, api_key=args.api_key)
    deadline = time.monotonic() + args.timeout_seconds
    last_error: Exception | None = None

    while time.monotonic() < deadline:
        try:
            models = client.models.list()
            model_ids = [model.id for model in models.data]
            if args.expected_model is None or args.expected_model in model_ids or model_ids:
                print(model_ids)
                return
        except (APIConnectionError, APITimeoutError) as exc:
            last_error = exc
        time.sleep(args.sleep_seconds)

    message = f"vLLM did not become ready within {args.timeout_seconds} seconds"
    if last_error is not None:
        raise TimeoutError(message) from last_error
    raise TimeoutError(message)


if __name__ == "__main__":
    main()
