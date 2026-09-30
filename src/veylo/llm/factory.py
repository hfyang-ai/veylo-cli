from __future__ import annotations

from veylo.config import LlmConfig
from veylo.llm.catalog import (
    MODELS,
    PROVIDERS,
    get_model_spec,
    get_provider,
    resolve_base_url,
    resolve_compat,
    resolve_context_window,
)
from veylo.llm.openai_compatible import OpenAICompatibleClient
from veylo.llm.pricing import resolve_price_profile

# ---------------------------------------------------------------------------
# Legacy constants — derived from the catalog and kept for backward
# compatibility (``entrypoints.repl.config_base_url`` still reads them). New
# code should read from ``veylo.llm.catalog`` directly.
# ---------------------------------------------------------------------------
DEEPSEEK_BASE_URL = PROVIDERS["deepseek"].base_url
OPENAI_BASE_URL = PROVIDERS["openai"].base_url
PROVIDER_BASE_URLS = {slug: spec.base_url for slug, spec in PROVIDERS.items()}
PROVIDER_BASE_URLS["zhipu"] = PROVIDERS["glm"].base_url
PROVIDER_BASE_URLS["moonshot"] = PROVIDERS["kimi"].base_url
MODEL_CONTEXT_WINDOWS = {spec.model: spec.context_window for spec in MODELS}


def create_llm_client(config: LlmConfig) -> OpenAICompatibleClient:
    """Build an :class:`OpenAICompatibleClient` from config.

    Provider base URL, context window and compatibility flags are resolved from
    the single catalog in ``veylo.llm.catalog`` instead of hand-written
    ``if provider == ...`` branches, so adding a provider is a data change.
    """
    provider = config.provider.lower()
    spec = get_model_spec(config.model)

    base_url = resolve_base_url(provider, spec, config.base_url) or DEEPSEEK_BASE_URL
    context = resolve_context_window(
        provider,
        spec,
        config.context_window,
        fallback=_context_fallback(provider),
    )
    compat = resolve_compat(provider, spec)
    include_builtin = spec is not None and spec.pricing == "builtin"

    return OpenAICompatibleClient(
        provider_name=provider,
        model=config.model,
        api_key=config.api_key,
        base_url=base_url,
        max_tokens=config.max_tokens,
        temperature=config.temperature,
        timeout=config.timeout,
        max_context_window=context,
        compat=compat,
        price_profile=resolve_price_profile(
            config.model,
            context_window=context,
            overrides=config.prices,
            include_builtin=include_builtin,
        ),
    )


def _context_fallback(provider: str) -> int:
    """Unknown-model context fallback matching the historical behaviour:
    DeepSeek (and unknown providers) default to 64k, other known providers 128k.
    """
    provider_spec = get_provider(provider)
    if provider_spec is not None and provider_spec.slug != "deepseek":
        return 128_000
    return 64_000
