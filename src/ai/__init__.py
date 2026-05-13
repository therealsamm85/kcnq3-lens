"""Optional AI interpretation across multiple LLM providers.

Supported providers (user supplies their own API key):
- Anthropic Claude
- OpenAI GPT
- Google Gemini
"""

from .router import (
    interpret_findings,
    interpret_comparison,
    list_providers,
    list_available_providers,
    get_provider_class,
    PROVIDERS,
)
from .prompt import (
    build_findings_payload,
    build_comparison_payload,
    SYSTEM_PROMPT,
    COMPARISON_SYSTEM_PROMPT,
)
from .base import LLMProvider, ProviderInfo

__all__ = [
    "interpret_findings",
    "interpret_comparison",
    "list_providers",
    "list_available_providers",
    "get_provider_class",
    "PROVIDERS",
    "build_findings_payload",
    "build_comparison_payload",
    "SYSTEM_PROMPT",
    "COMPARISON_SYSTEM_PROMPT",
    "LLMProvider",
    "ProviderInfo",
]
