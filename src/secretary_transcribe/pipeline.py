from __future__ import annotations

import tempfile
from pathlib import Path

from pydantic import BaseModel

from secretary_transcribe import audio, transcribe, translate
from secretary_transcribe.config import DEFAULT_TRANSLATION_CONTEXT, get_settings


class PipelineResult(BaseModel):
    duration_seconds: float
    afrikaans: str
    english: str
    model: str


def _extract(response: object, attr: str, default: object = None) -> object:
    if isinstance(response, dict):
        return response.get(attr, default)
    return getattr(response, attr, default)


async def run_pipeline(
    audio_path: Path,
    *,
    cheap: bool = False,
    context: str | None = None,
) -> PipelineResult:
    """Run the full pipeline: ffmpeg-normalize -> Whisper(af) -> GPT translate(en)."""
    settings = get_settings()
    whisper_prompt = settings.whisper_prompt
    translation_context = context if context is not None else DEFAULT_TRANSLATION_CONTEXT

    with tempfile.TemporaryDirectory() as tmp:
        temp_wav = Path(tmp) / "audio.wav"
        audio.normalize_to_wav(audio_path, temp_wav)

        response = await transcribe.transcribe_afrikaans(temp_wav, prompt=whisper_prompt)

    text = (_extract(response, "text", "") or "")
    duration_raw = _extract(response, "duration", 0.0)
    try:
        duration = float(duration_raw) if duration_raw is not None else 0.0
    except (TypeError, ValueError):
        duration = 0.0

    if not text.strip():
        raise RuntimeError("No speech detected in audio (empty transcription)")

    english, model_name = await translate.translate_to_english(
        text, context=translation_context, cheap=cheap
    )

    return PipelineResult(
        duration_seconds=duration,
        afrikaans=text.strip(),
        english=english.strip(),
        model=model_name,
    )
