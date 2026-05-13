"""Unified entry point for AI interpretation across providers."""

from __future__ import annotations

from typing import Any

from .base import LLMProvider, ProviderInfo
from .prompt import (
    SYSTEM_PROMPT,
    COMPARISON_SYSTEM_PROMPT,
    build_findings_payload,
    build_comparison_payload,
    build_user_message,
)
from .providers import AnthropicProvider, OpenAIProvider, GeminiProvider


PROVIDERS: dict[str, type[LLMProvider]] = {
    AnthropicProvider.info.id: AnthropicProvider,
    OpenAIProvider.info.id: OpenAIProvider,
    GeminiProvider.info.id: GeminiProvider,
}


def list_providers() -> list[ProviderInfo]:
    """Return metadata for every registered provider (for UI dropdowns)."""
    return [cls.info for cls in PROVIDERS.values()]


def list_available_providers() -> list[ProviderInfo]:
    """Return only providers whose Python SDK is currently installed."""
    return [cls.info for cls in PROVIDERS.values() if cls.is_available()]


def get_provider_class(provider_id: str) -> type[LLMProvider]:
    if provider_id not in PROVIDERS:
        raise ValueError(
            f"Unknown provider '{provider_id}'. Available: {list(PROVIDERS.keys())}"
        )
    return PROVIDERS[provider_id]


def _make_provider(provider_id: str, api_key: str, model: str | None) -> LLMProvider:
    cls = get_provider_class(provider_id)
    if not cls.is_available():
        raise RuntimeError(
            f"{cls.info.display_name} SDK is not installed. "
            f"Install with: pip install {cls.info.pip_package}"
        )
    return cls(api_key=api_key, model=model)


def interpret_findings(
    provider_id: str,
    api_key: str,
    findings: dict[str, Any],
    age_years: float | None = None,
    variant: str | None = None,
    model: str | None = None,
    max_tokens: int = 2000,
) -> str:
    """Generate a plain-language interpretation of single-recording findings."""
    provider = _make_provider(provider_id, api_key, model)
    payload = build_findings_payload(findings, age_years=age_years, variant=variant)
    user_message = build_user_message(payload, task="single")
    return provider.interpret(SYSTEM_PROMPT, user_message, max_tokens=max_tokens)


def interpret_comparison(
    provider_id: str,
    api_key: str,
    comparison: dict[str, Any],
    age_years: float | None = None,
    variant: str | None = None,
    pre_label: str = "pre-treatment",
    post_label: str = "post-treatment",
    model: str | None = None,
    max_tokens: int = 2500,
) -> str:
    """Generate a plain-language interpretation of pre/post comparison."""
    provider = _make_provider(provider_id, api_key, model)
    payload = build_comparison_payload(
        comparison,
        age_years=age_years,
        variant=variant,
        pre_label=pre_label,
        post_label=post_label,
    )
    user_message = build_user_message(payload, task="compare")
    return provider.interpret(COMPARISON_SYSTEM_PROMPT, user_message, max_tokens=max_tokens)
