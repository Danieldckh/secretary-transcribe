from __future__ import annotations

import tempfile
from pathlib import Path

from pydantic import BaseModel

from secretary_transcribe import audio, transcribe, translate
from secretary_transcribe.config import DEFAULT_TRANSLATION_CONTEXT, get_settings

CHUNK_DURATION_SECONDS = 600
OVERLAP_SECONDS = 5
WHISPER_SAFE_BYTES = 22 * 1024 * 1024


class PipelineResult(BaseModel):
    duration_seconds: float
    afrikaans: str
    english: str
    model: str


def _extract(response: object, attr: str, default: object = None) -> object:
    if isinstance(response, dict):
        return response.get(attr, default)
    return getattr(response, attr, default)


async def _transcribe_chunks(chunks: list[Path], *, prompt: str) -> str:
    """Transcribe each chunk sequentially and join the Afrikaans text with single spaces."""
    parts: list[str] = []
    for chunk in chunks:
        response = await transcribe.transcribe_afrikaans(chunk, prompt=prompt)
        text = (_extract(response, "text", "") or "").strip()
        if text:
            parts.append(text)
    return " ".join(parts)


async def run_pipeline(
    audio_path: Path,
    *,
    cheap: bool = False,
    context: str | None = None,
) -> PipelineResult:
    """Run the pipeline: ffmpeg -> opus -> (chunk if large) -> Whisper(af) -> GPT translate."""
    settings = get_settings()
    whisper_prompt = settings.whisper_prompt
    translation_context = context if context is not None else DEFAULT_TRANSLATION_CONTEXT

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        normalized = tmp_dir / "audio.ogg"
        audio.normalize_to_opus(audio_path, normalized)

        duration = audio.get_duration_seconds(normalized)
        size = normalized.stat().st_size

        if size <= WHISPER_SAFE_BYTES and duration <= CHUNK_DURATION_SECONDS:
            chunks = [normalized]
        else:
            chunks = audio.split_into_chunks(
                normalized,
                tmp_dir,
                chunk_seconds=CHUNK_DURATION_SECONDS,
                overlap_seconds=OVERLAP_SECONDS,
            )

        combined_afrikaans = await _transcribe_chunks(chunks, prompt=whisper_prompt)

    if not combined_afrikaans.strip():
        raise RuntimeError("No speech detected in audio (empty transcription)")

    cleaned_af, model_name = await translate.fix_afrikaans_construction(
        combined_afrikaans, context=translation_context, cheap=cheap
    )
    english, _ = await translate.translate_to_english(
        cleaned_af, context=translation_context, cheap=cheap
    )

    return PipelineResult(
        duration_seconds=duration,
        afrikaans=cleaned_af,
        english=english,
        model=model_name,
    )
