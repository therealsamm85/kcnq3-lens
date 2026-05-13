"""Anthropic Claude provider."""

from __future__ import annotations

from ..base import LLMProvider, ProviderInfo


class AnthropicProvider(LLMProvider):

    info = ProviderInfo(
        id="anthropic",
        display_name="Claude (Anthropic)",
        default_model="claude-opus-4-7",
        available_models=[
            "claude-opus-4-7",
            "claude-sonnet-4-7",
            "claude-haiku-4-7",
        ],
        api_key_url="https://console.anthropic.com/settings/keys",
        pip_package="anthropic",
    )

    def interpret(self, system_prompt: str, user_message: str, max_tokens: int = 2000) -> str:
        from anthropic import Anthropic

        client = Anthropic(api_key=self.api_key)
        response = client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}],
        )
        return response.content[0].text

    @staticmethod
    def _import_sdk():
        import anthropic  # noqa: F401
