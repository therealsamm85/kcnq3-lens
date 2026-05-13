"""OpenAI GPT provider."""

from __future__ import annotations

from ..base import LLMProvider, ProviderInfo


class OpenAIProvider(LLMProvider):

    info = ProviderInfo(
        id="openai",
        display_name="GPT (OpenAI)",
        default_model="gpt-4o",
        available_models=[
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-4-turbo",
            "o1-mini",
        ],
        api_key_url="https://platform.openai.com/api-keys",
        pip_package="openai",
    )

    def interpret(self, system_prompt: str, user_message: str, max_tokens: int = 2000) -> str:
        from openai import OpenAI

        client = OpenAI(api_key=self.api_key)
        response = client.chat.completions.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
        )
        return response.choices[0].message.content

    @staticmethod
    def _import_sdk():
        import openai  # noqa: F401
