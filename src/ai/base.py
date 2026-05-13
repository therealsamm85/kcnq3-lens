"""Abstract base class for LLM providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class ProviderInfo:
    """Static metadata for a provider — used in UI dropdowns."""

    id: str                 # short id, e.g. "anthropic"
    display_name: str       # "Claude (Anthropic)"
    default_model: str
    available_models: list[str]
    api_key_url: str        # where the user goes to get a key
    pip_package: str        # what they need installed


class LLMProvider(ABC):
    """Unified interface for any LLM provider."""

    info: ProviderInfo

    def __init__(self, api_key: str, model: str | None = None):
        if not api_key:
            raise ValueError(f"{self.info.display_name}: API key is required.")
        self.api_key = api_key
        self.model = model or self.info.default_model

    @abstractmethod
    def interpret(self, system_prompt: str, user_message: str, max_tokens: int = 2000) -> str:
        """Send the prompt to the provider and return the generated text."""
        raise NotImplementedError

    @classmethod
    def is_available(cls) -> bool:
        """Check whether the provider's Python SDK is installed."""
        try:
            cls._import_sdk()
            return True
        except ImportError:
            return False

    @staticmethod
    @abstractmethod
    def _import_sdk():
        """Import the SDK module — used for availability check."""
        raise NotImplementedError
