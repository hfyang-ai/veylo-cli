from __future__ import annotations

import asyncio

from veylo.config import LlmConfig
from veylo.llm import create_llm_client
from veylo.llm.catalog import (
    get_model_spec,
    get_provider,
    resolve_compat,
    resolve_context_window,
)
from veylo.llm.compat import OpenAICompat
from veylo.llm.openai_compatible import OpenAICompatibleClient
from veylo.types import Message


def _client(**kwargs) -> OpenAICompatibleClient:
    defaults = dict(
        provider_name="deepseek",
        model="deepseek-v4-flash",
        api_key="key",
        base_url="https://api.deepseek.com/v1",
    )
    defaults.update(kwargs)
    return OpenAICompatibleClient(**defaults)


# --- OpenAICompat request-payload switches ---


def test_compat_defaults_preserve_openai_behaviour() -> None:
    payload = _client()._build_payload([Message(role="user", content="hi")], [], system_prompt="s")
    assert "max_tokens" in payload
    assert payload["stream_options"] == {"include_usage": True}


def test_compat_max_completion_tokens_field() -> None:
    client = _client(compat=OpenAICompat(max_tokens_field="max_completion_tokens"))
    payload = client._build_payload([], [], system_prompt="s")
    assert "max_completion_tokens" in payload
    assert "max_tokens" not in payload


def test_compat_omits_stream_options_when_disabled() -> None:
    client = _client(compat=OpenAICompat(include_stream_options=False))
    payload = client._build_payload([], [], system_prompt="s")
    assert "stream_options" not in payload


def test_compat_custom_thinking_field_is_parsed() -> None:
    client = _client(compat=OpenAICompat(thinking_field="thinking"))
    events = asyncio.run(_collect(client, {"choices": [{"delta": {"thinking": "hmm"}}]}))
    thinking = [e["thinking"] for e in events if e["type"] == "thinking_delta"]
    assert thinking == ["hmm"]


def test_compat_tool_result_requires_name() -> None:
    messages = [Message(role="tool", content="result", tool_call_id="call_1", name="read_file")]
    with_name = _client(compat=OpenAICompat(tool_result_requires_name=True))
    tool_msg = [m for m in with_name._format_messages(messages, "s") if m["role"] == "tool"][0]
    assert tool_msg["name"] == "read_file"

    default_msg = [m for m in _client()._format_messages(messages, "s") if m["role"] == "tool"][0]
    assert "name" not in default_msg


def test_compat_supports_images_override() -> None:
    assert _client(compat=OpenAICompat(supports_images=True)).supports_images is True
    assert _client(compat=OpenAICompat(supports_images=False)).supports_images is False
    # default: inferred from model name (deepseek-v4-flash has no vision marker)
    assert _client().supports_images is False


# --- catalog resolution ---


def test_provider_aliases_resolve() -> None:
    assert get_provider("zhipu").slug == "glm"
    assert get_provider("moonshot").slug == "kimi"
    assert get_provider("openai-compatible").slug == "openai"
    assert get_provider("qwen").slug == "qwen"


def test_model_spec_lookup_is_case_insensitive() -> None:
    assert get_model_spec("QWEN-MAX").model == "qwen-max"
    assert get_model_spec("deepseek-v4-flash").context_window == 1_000_000


def test_resolve_context_window_priority() -> None:
    # explicit config wins
    assert resolve_context_window("qwen", None, 64_000) == 64_000
    # model spec wins over provider default
    spec = get_model_spec("deepseek-coder")
    assert resolve_context_window("deepseek", spec, None) == 128_000
    # provider default when model unknown
    assert resolve_context_window("glm", None, None) == 200_000


def test_create_client_for_qwen_resolves_dashscope() -> None:
    client = create_llm_client(LlmConfig(provider="qwen", model="qwen-max"))
    assert client.base_url == "https://dashscope.aliyuncs.com/compatible-mode/v1"
    assert client.max_context_window == 128_000
    assert client.compat.prompt_cache is False


def test_create_client_deepseek_keeps_prompt_cache_and_builtin_prices() -> None:
    client = create_llm_client(LlmConfig(provider="deepseek", model="deepseek-v4-flash"))
    assert client.compat.prompt_cache is True
    assert client.price_profile is not None


async def _collect(client: OpenAICompatibleClient, chunk: dict) -> list[dict]:
    return [event async for event in client._parse_chunk(chunk)]
