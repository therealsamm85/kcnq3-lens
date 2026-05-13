"""Internationalization (i18n) for KCNQ3-Lens.

Lightweight key-based translator. English is the source of truth; missing
keys in other languages fall back to English so the app never breaks.
"""

from __future__ import annotations

from .translations import TRANSLATIONS

LANGUAGES = {
    "en": "English",
    "de": "Deutsch",
}


class Translator:
    """Simple key-based translator with format-string substitution."""

    def __init__(self, language: str = "en"):
        self.set_language(language)

    def set_language(self, language: str) -> None:
        if language not in TRANSLATIONS:
            language = "en"
        self.language = language
        self._strings = TRANSLATIONS[language]
        self._fallback = TRANSLATIONS["en"]

    def t(self, key: str, **kwargs) -> str:
        """Return translated string. Falls back to English, then to key itself."""
        text = self._strings.get(key) or self._fallback.get(key) or key
        try:
            return text.format(**kwargs) if kwargs else text
        except (KeyError, IndexError):
            return text


def get_translator(language: str = "en") -> Translator:
    return Translator(language)


__all__ = ["Translator", "get_translator", "LANGUAGES", "TRANSLATIONS"]
