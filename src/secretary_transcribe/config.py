from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None


DEFAULT_WHISPER_PROMPT = (
    "Hierdie is 'n WhatsApp-stemboodskap van 'n Suid-Afrikaanse boer of plaasbestuurder. "
    "Algemene woorde sluit in: stoppelland, plaasbestuurder, oesseisoen, "
    "natbalblad-besproeiing, pesbestuur, hoenderbatterye, melkery, mielies, koring, "
    "sojabone, sonneblom, lusern, beeste, skape, bokke, varke, trekker, stoorkamer, "
    "graansilo, kontrakteur, weiding, kraal, voerkraal, droogte, reënval."
)

DEFAULT_TRANSLATION_CONTEXT = (
    "This is a casual WhatsApp voice note from a South African farmer or farm manager, "
    "often discussing crops, livestock, weather, suppliers, or daily farm operations."
)


@dataclass(frozen=True)
class Settings:
    openai_api_key: str
    whisper_prompt: str
    translation_context: str


_loaded_dotenv = False


def _ensure_dotenv_loaded() -> None:
    global _loaded_dotenv
    if _loaded_dotenv or load_dotenv is None:
        return
    for candidate in (Path.cwd() / ".env", Path(__file__).resolve().parents[2] / ".env"):
        if candidate.exists():
            load_dotenv(candidate, override=False)
            break
    _loaded_dotenv = True


def get_settings() -> Settings:
    _ensure_dotenv_loaded()
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")
    return Settings(
        openai_api_key=api_key,
        whisper_prompt=DEFAULT_WHISPER_PROMPT,
        translation_context=DEFAULT_TRANSLATION_CONTEXT,
    )
