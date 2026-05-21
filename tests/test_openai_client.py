from types import SimpleNamespace

from spider4ssc_zeroshot.openai_client import OpenAIChatClient, OpenAIChatClientConfig


class _Choice:
    finish_reason = "stop"
    message = SimpleNamespace(content="SELECT 1")


class _Usage:
    prompt_tokens = 2
    completion_tokens = 3
    total_tokens = 5


class _CompletionsEndpoint:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def create(self, **kwargs) -> SimpleNamespace:
        self.calls.append(kwargs)
        return SimpleNamespace(
            choices=[_Choice()],
            usage=_Usage(),
            model="gpt-5.4-mini-2026-03-17",
        )


class _ChatEndpoint:
    def __init__(self) -> None:
        self.completions = _CompletionsEndpoint()


class _FakeOpenAI:
    def __init__(self) -> None:
        self.chat = _ChatEndpoint()


def _client_with_fake_openai() -> OpenAIChatClient:
    client = OpenAIChatClient(
        OpenAIChatClientConfig(
            base_url="https://api.openai.com/v1",
            api_key="sk-test",
            request_timeout_seconds=1,
            max_retries=0,
            retry_sleep_seconds=0,
        )
    )
    client._client = _FakeOpenAI()
    return client


def test_openai_chat_client_sends_chat_completion_with_reasoning_effort():
    client = _client_with_fake_openai()

    result = client.complete(
        "prompt",
        "gpt-5.4-mini-2026-03-17",
        {
            "temperature": 0.0,
            "top_p": 1.0,
            "max_completion_tokens": 17,
            "stop": ["```"],
            "reasoning_effort": "none",
        },
    )

    assert result == {
        "raw_completion": "SELECT 1",
        "finish_reason": "stop",
        "usage": {
            "prompt_tokens": 2,
            "completion_tokens": 3,
            "total_tokens": 5,
        },
        "model_revision": "gpt-5.4-mini-2026-03-17",
    }
    assert client._client.chat.completions.calls == [
        {
            "model": "gpt-5.4-mini-2026-03-17",
            "messages": [{"role": "user", "content": "prompt"}],
            "temperature": 0.0,
            "top_p": 1.0,
            "max_completion_tokens": 17,
            "stop": ["```"],
            "store": False,
            "reasoning_effort": "none",
        }
    ]


def test_openai_chat_client_omits_reasoning_effort_when_not_configured():
    client = _client_with_fake_openai()

    client.complete(
        "prompt",
        "gpt-5.4-mini-2026-03-17",
        {
            "temperature": 0.0,
            "top_p": 1.0,
            "max_completion_tokens": 17,
            "stop": [],
            "reasoning_effort": None,
        },
    )

    assert "reasoning_effort" not in client._client.chat.completions.calls[0]
