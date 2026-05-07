from __future__ import annotations

from pathlib import Path
from typing import Any

from openai import AsyncOpenAI


async def transcribe_afrikaans(audio_path: Path, *, prompt: str) -> Any:
    """Transcribe Afrikaans audio with Whisper, returning the verbose_json response."""
    client = AsyncOpenAI()
    audio_bytes = audio_path.read_bytes()
    response = await client.audio.transcriptions.create(
        file=(audio_path.name or "audio.wav", audio_bytes, "audio/wav"),
        model="whisper-1",
        language="af",
        prompt=prompt,
        response_format="verbose_json",
    )
    return response
