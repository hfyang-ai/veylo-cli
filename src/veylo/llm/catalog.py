"""catalog.py — Single source of truth for providers and known models.

Before this module, model metadata lived in four places that had to be kept in
sync by hand: ``factory.MODEL_CONTEXT_WINDOWS``, ``factory.PROVIDER_BASE_URLS``,
``model_profiles.DEFAULT_MODEL_PROFILES`` and ``pricing.DEEPSEEK_V4_PRICE_PROFILES``.
``models.py`` even carried a "Kept in sync with..." comment as evidence of the
drift risk.

``PROVIDERS`` and ``MODELS`` are now the canonical data. Every derived surface
(factory defaults, the interactive model selector, ``/model list``, context
windows, base URLs, API-key env names) is generated from these two tables so a
change is made once and propagates everywhere.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from veylo.llm.compat import (
    DEEPSEEK_COMPAT,
    GLM_COMPAT,
    KIMI_COMPAT,
    OPENAI_COMPAT,
    QWEN_COMPAT,
    OpenAICompat,
)


@dataclass(frozen=True, slots=True)
class ProviderSpec:
    """Provider-level defaults: base URL, context window, auth and compat."""

    slug: str
    name: str
    base_url: str
    default_context_window: int
    api_key_envs: tuple[str, ...] = ()
    compat: OpenAICompat = field(default_factory=OpenAICompat)


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """A single known model entry."""

    provider: str
    model: str
    name: str
    context_window: int
    pricing: Literal["builtin", "unknown"] = "unknown"
    description: str = ""
    base_url: str | None = None
    compat: OpenAICompat | None = None
    default_profile: bool = False
    """Whether the model shows up in the interactive default-model selector."""


# ---------------------------------------------------------------------------
# Providers
# ---------------------------------------------------------------------------

PROVIDERS: dict[str, ProviderSpec] = {
    "deepseek": ProviderSpec(
        slug="deepseek",
        name="DeepSeek",
        base_url="https://api.deepseek.com/v1",
        default_context_window=1_000_000,
        api_key_envs=("DEEPSEEK_API_KEY",),
        compat=DEEPSEEK_COMPAT,
    ),
    "openai": ProviderSpec(
        slug="openai",
        name="OpenAI",
        base_url="https://api.openai.com/v1",
        default_context_window=128_000,
        api_key_envs=("OPENAI_API_KEY",),
        compat=OPENAI_COMPAT,
    ),
    "glm": ProviderSpec(
        slug="glm",
        name="GLM / Zhipu",
        base_url="https://open.bigmodel.cn/api/paas/v4",
        default_context_window=200_000,
        api_key_envs=("ZAI_API_KEY", "GLM_API_KEY"),
        compat=GLM_COMPAT,
    ),
    "kimi": ProviderSpec(
        slug="kimi",
        name="Kimi / Moonshot",
        base_url="https://api.moonshot.cn/v1",
        default_context_window=128_000,
        api_key_envs=("KIMI_API_KEY",),
        compat=KIMI_COMPAT,
    ),
    "qwen": ProviderSpec(
        slug="qwen",
        name="Qwen / DashScope",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        default_context_window=128_000,
        api_key_envs=("DASHSCOPE_API_KEY",),
        compat=QWEN_COMPAT,
    ),
}

# Provider slugs users may type that map onto a canonical provider.
PROVIDER_ALIASES: dict[str, str] = {
    "zhipu": "glm",
    "moonshot": "kimi",
    "openai-compatible": "openai",
    "compatible": "openai",
}


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

MODELS: tuple[ModelSpec, ...] = (
    # ---- DeepSeek ----
    ModelSpec(
        provider="deepseek",
        model="deepseek-v4-flash",
        name="DeepSeek V4 Flash",
        context_window=1_000_000,
        pricing="builtin",
        description="Fast, cost-efficient Agent model with thinking support",
        default_profile=True,
    ),
    ModelSpec(
        provider="deepseek",
        model="deepseek-v4-pro",
        name="DeepSeek V4 Pro",
        context_window=1_000_000,
        pricing="builtin",
        description="Higher-quality DeepSeek model for difficult coding tasks",
        default_profile=True,
    ),
    ModelSpec(
        provider="deepseek",
        model="deepseek-chat",
        name="DeepSeek Chat",
        context_window=1_000_000,
    ),
    ModelSpec(
        provider="deepseek",
        model="deepseek-reasoner",
        name="DeepSeek Reasoner",
        context_window=1_000_000,
    ),
    ModelSpec(
        provider="deepseek",
        model="deepseek-coder",
        name="DeepSeek Coder",
        context_window=128_000,
    ),
    # ---- OpenAI ----
    ModelSpec(
        provider="openai",
        model="gpt-4o",
        name="GPT-4o",
        context_window=128_000,
    ),
    ModelSpec(
        provider="openai",
        model="gpt-4o-mini",
        name="GPT-4o mini",
        context_window=128_000,
    ),
    # ---- GLM / Zhipu ----
    ModelSpec(
        provider="glm",
        model="glm-5.2",
        name="GLM-5.2",
        context_window=200_000,
        description="Zhipu flagship model for long-running Agent tasks",
        default_profile=True,
    ),
    ModelSpec(
        provider="glm",
        model="glm-5.1",
        name="GLM-5.1",
        context_window=200_000,
        description="Zhipu general-purpose coding and reasoning model",
        default_profile=True,
    ),
    ModelSpec(
        provider="glm",
        model="glm-4.7",
        name="GLM-4.7",
        context_window=200_000,
        description="Agentic coding model with tool calling",
        default_profile=True,
    ),
    # ---- Kimi / Moonshot ----
    ModelSpec(
        provider="kimi",
        model="kimi-think",
        name="Kimi Think",
        context_window=128_000,
    ),
    ModelSpec(
        provider="moonshot",
        model="moonshot-v1-8k",
        name="Moonshot v1 8k",
        context_window=8_000,
    ),
    ModelSpec(
        provider="moonshot",
        model="moonshot-v1-32k",
        name="Moonshot v1 32k",
        context_window=32_000,
    ),
    ModelSpec(
        provider="moonshot",
        model="moonshot-v1-128k",
        name="Moonshot v1 128k",
        context_window=128_000,
    ),
    # ---- Qwen / DashScope ----
    ModelSpec(
        provider="qwen",
        model="qwen-max",
        name="Qwen Max",
        context_window=128_000,
    ),
    ModelSpec(
        provider="qwen",
        model="qwen-plus",
        name="Qwen Plus",
        context_window=128_000,
    ),
)


# ---------------------------------------------------------------------------
# Lookup + resolution helpers
# ---------------------------------------------------------------------------

def get_provider(slug: str) -> ProviderSpec | None:
    """Resolve a provider slug (aliases included) to its :class:`ProviderSpec`."""
    canonical = PROVIDER_ALIASES.get(slug.lower(), slug.lower())
    return PROVIDERS.get(canonical)


def get_model_spec(model: str) -> ModelSpec | None:
    """Resolve a model identifier (case-insensitive) to its :class:`ModelSpec`."""
    model_lower = model.lower()
    for spec in MODELS:
        if spec.model.lower() == model_lower:
            return spec
    return None


def resolve_compat(provider: str, spec: ModelSpec | None) -> OpenAICompat:
    """Pick the effective compat for a provider/model, model overriding provider."""
    if spec is not None and spec.compat is not None:
        return spec.compat
    provider_spec = get_provider(provider)
    if provider_spec is not None:
        return provider_spec.compat
    return OpenAICompat()


def resolve_base_url(
    provider: str,
    spec: ModelSpec | None,
    explicit: str | None,
) -> str | None:
    """Resolve the request base URL: explicit config > model override > provider."""
    if explicit:
        return explicit
    if spec is not None and spec.base_url:
        return spec.base_url
    provider_spec = get_provider(provider)
    if provider_spec is not None:
        return provider_spec.base_url
    return None


def resolve_context_window(
    provider: str,
    spec: ModelSpec | None,
    explicit: int | None,
    fallback: int = 64_000,
) -> int:
    """Resolve the context window: explicit config > model > provider > fallback."""
    if explicit:
        return explicit
    if spec is not None:
        return spec.context_window
    provider_spec = get_provider(provider)
    if provider_spec is not None:
        return provider_spec.default_context_window
    return fallback
