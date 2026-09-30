"""models.py — Central registry of known models for display and selection.

Provides a curated built-in list of models aggregated from across the codebase
(catalog context windows, pricing profiles, known providers) plus a function
to retrieve the full list.  Users can always configure arbitrary models via
LlmConfig; this module only surfaces the ones Veylo knows about.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from veylo.llm.catalog import MODELS


@dataclass(frozen=True, slots=True)
class KnownModel:
    """A single known model entry for display in ``/model list``."""

    provider: str
    """Provider slug, e.g. ``"deepseek"``, ``"openai"``."""

    model: str
    """Full model identifier as passed to the API, e.g. ``"deepseek-v4-flash"``."""

    context_window: int
    """Maximum context window in tokens."""

    pricing: Literal["builtin", "unknown"] = "unknown"
    """Whether built-in price data is available for this model."""


# ---------------------------------------------------------------------------
# Curated built-in list
# ---------------------------------------------------------------------------
# Derived from the single catalog in ``veylo.llm.catalog`` so context windows,
# providers and pricing flags can't drift out of sync with the factory or the
# model selector.

_BUILTIN_MODELS: list[KnownModel] = [
    KnownModel(
        provider=spec.provider,
        model=spec.model,
        context_window=spec.context_window,
        pricing=spec.pricing,
    )
    for spec in MODELS
]


def get_available_models() -> list[KnownModel]:
    """Return the curated list of known models.

    This list is the single source of truth for ``/model list`` display.
    It is safe to call without any API key or network access.
    """
    return list(_BUILTIN_MODELS)


def get_models_by_provider(provider: str) -> list[KnownModel]:
    """Filter :func:`get_available_models` by provider slug (case-insensitive)."""
    provider_lower = provider.lower()
    return [m for m in _BUILTIN_MODELS if m.provider.lower() == provider_lower]
