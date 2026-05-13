"""Google Gemini provider."""

from __future__ import annotations

from ..base import LLMProvider, ProviderInfo


class GeminiProvider(LLMProvider):

    info = ProviderInfo(
        id="gemini",
        display_name="Gemini (Google)",
        default_model="gemini-1.5-pro",
        available_models=[
            "gemini-1.5-pro",
            "gemini-1.5-flash",
            "gemini-2.0-flash-exp",
        ],
        api_key_url="https://aistudio.google.com/app/apikey",
        pip_package="google-generativeai",
    )

    def interpret(self, system_prompt: str, user_message: str, max_tokens: int = 2000) -> str:
        import google.generativeai as genai

        genai.configure(api_key=self.api_key)
        model = genai.GenerativeModel(
            model_name=self.model,
            system_instruction=system_prompt,
        )
        response = model.generate_content(
            user_message,
            generation_config={
                "max_output_tokens": max_tokens,
                "temperature": 0.7,
            },
        )
        return response.text

    @staticmethod
    def _import_sdk():
        import google.generativeai  # noqa: F401
