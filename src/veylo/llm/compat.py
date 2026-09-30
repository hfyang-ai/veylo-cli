"""compat.py — Declarative per-provider quirks for OpenAI-compatible endpoints.

Not every provider that exposes an OpenAI-style ``/chat/completions`` endpoint is
100% compatible. The differences are small but real, and historically got
hard-coded as scattered ``if`` branches (or just silently broke). ``OpenAICompat``
groups them into a single frozen dataclass so each provider can describe its
deviations in one place, and the client reads from it instead of guessing.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class OpenAICompat:
    """Compatibility flags for an OpenAI-compatible provider.

    Every field has a default that matches the canonical OpenAI behaviour, so
    ``OpenAICompat()`` describes a "well-behaved" provider and providers only
    override what they actually deviate on.
    """

    # --- request payload shape ---
    max_tokens_field: str = "max_tokens"
    """Request field for the token cap. Newer OpenAI o-series models require
    ``"max_completion_tokens"``; most others accept ``"max_tokens"``."""

    include_stream_options: bool = True
    """Whether to send ``stream_options={"include_usage": True}``. Some private
    gateways 400 on this field, so they can turn it off."""

    prompt_cache: bool = False
    """Whether the provider supports prompt caching (DeepSeek does)."""

    # --- thinking / reasoning ---
    thinking_field: str = "reasoning_content"
    """Response delta field that carries the model's reasoning text. DeepSeek,
    Zhipu, Kimi and Qwen all use ``reasoning_content``; alternatives use
    ``reasoning`` or ``thinking``."""

    # --- message formatting ---
    tool_result_requires_name: bool = False
    """Whether a ``tool`` role message must also carry a ``name`` field (some
    OpenAI-compatible gateways expect it alongside ``tool_call_id``)."""

    # --- images ---
    supports_images: bool | None = None
    """Whether the model accepts image parts. ``None`` means infer from the model
    name (e.g. ``vision``/``5v``/``vl`` markers)."""


# Prebuilt profiles for known providers. These are the single place to record a
# provider's deviations; the model catalog and factory both consume them.
DEEPSEEK_COMPAT = OpenAICompat(prompt_cache=True)
"""DeepSeek supports prompt caching and otherwise follows OpenAI conventions."""

GLM_COMPAT = OpenAICompat()
"""Zhipu GLM is OpenAI-compatible and reports reasoning via ``reasoning_content``."""

OPENAI_COMPAT = OpenAICompat()
"""Canonical OpenAI behaviour (o-series ``max_completion_tokens`` is opt-in)."""

KIMI_COMPAT = OpenAICompat()
"""Kimi / Moonshot follows OpenAI conventions; reasoning via ``reasoning_content``."""

QWEN_COMPAT = OpenAICompat()
"""Qwen (DashScope) follows OpenAI conventions; reasoning via ``reasoning_content``."""
