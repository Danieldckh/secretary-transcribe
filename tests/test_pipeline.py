from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from secretary_transcribe import audio
from secretary_transcribe.audio import FfmpegFailed, FfmpegMissing, normalize_to_wav
from secretary_transcribe.pipeline import PipelineResult, run_pipeline


@pytest.fixture(autouse=True)
def _set_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test")


def _fake_whisper_response(text: str = "Hallo wereld", duration: float = 4.2) -> SimpleNamespace:
    return SimpleNamespace(text=text, duration=duration, segments=[])


@pytest.mark.asyncio
async def test_run_pipeline_happy_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "secretary_transcribe.audio.normalize_to_wav", lambda *a, **k: None
    )
    transcribe_mock = AsyncMock(return_value=_fake_whisper_response("Hallo wereld", 4.2))
    monkeypatch.setattr(
        "secretary_transcribe.transcribe.transcribe_afrikaans", transcribe_mock
    )
    translate_mock = AsyncMock(return_value=("Hello world", "gpt-4o"))
    monkeypatch.setattr(
        "secretary_transcribe.translate.translate_to_english", translate_mock
    )

    result = await run_pipeline(Path("fake.opus"))

    assert isinstance(result, PipelineResult)
    assert result.afrikaans == "Hallo wereld"
    assert result.english == "Hello world"
    assert result.duration_seconds == pytest.approx(4.2)
    assert result.model == "gpt-4o"
    transcribe_mock.assert_awaited_once()
    translate_mock.assert_awaited_once()


@pytest.mark.asyncio
async def test_run_pipeline_cheap_uses_mini(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "secretary_transcribe.audio.normalize_to_wav", lambda *a, **k: None
    )
    monkeypatch.setattr(
        "secretary_transcribe.transcribe.transcribe_afrikaans",
        AsyncMock(return_value=_fake_whisper_response("Hallo", 1.0)),
    )
    translate_mock = AsyncMock(return_value=("Hello", "gpt-4o-mini"))
    monkeypatch.setattr(
        "secretary_transcribe.translate.translate_to_english", translate_mock
    )

    result = await run_pipeline(Path("fake.opus"), cheap=True)

    assert result.model == "gpt-4o-mini"
    _, kwargs = translate_mock.await_args
    assert kwargs.get("cheap") is True


@pytest.mark.asyncio
async def test_run_pipeline_custom_context(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "secretary_transcribe.audio.normalize_to_wav", lambda *a, **k: None
    )
    monkeypatch.setattr(
        "secretary_transcribe.transcribe.transcribe_afrikaans",
        AsyncMock(return_value=_fake_whisper_response("Sitrus uitvoer", 2.0)),
    )
    translate_mock = AsyncMock(return_value=("Citrus exports", "gpt-4o"))
    monkeypatch.setattr(
        "secretary_transcribe.translate.translate_to_english", translate_mock
    )

    custom = "discussion about citrus exports"
    result = await run_pipeline(Path("fake.opus"), context=custom)

    assert result.english == "Citrus exports"
    _, kwargs = translate_mock.await_args
    assert kwargs.get("context") == custom


@pytest.mark.asyncio
async def test_run_pipeline_empty_transcription_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "secretary_transcribe.audio.normalize_to_wav", lambda *a, **k: None
    )
    monkeypatch.setattr(
        "secretary_transcribe.transcribe.transcribe_afrikaans",
        AsyncMock(return_value=_fake_whisper_response("   \n\t", 0.5)),
    )
    translate_mock = AsyncMock(return_value=("should not be called", "gpt-4o"))
    monkeypatch.setattr(
        "secretary_transcribe.translate.translate_to_english", translate_mock
    )

    with pytest.raises(RuntimeError, match="No speech"):
        await run_pipeline(Path("fake.opus"))

    translate_mock.assert_not_awaited()


def test_normalize_to_wav_missing_ffmpeg(monkeypatch: pytest.MonkeyPatch) -> None:
    def _raise(*args: object, **kwargs: object) -> None:
        raise FileNotFoundError("ffmpeg not found")

    monkeypatch.setattr(audio.subprocess, "run", _raise)

    with pytest.raises(FfmpegMissing):
        normalize_to_wav(Path("in.opus"), Path("out.wav"))


def test_normalize_to_wav_ffmpeg_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def _raise(*args: object, **kwargs: object) -> None:
        raise subprocess.CalledProcessError(
            returncode=1,
            cmd=["ffmpeg", "-i", "in.opus"],
            stderr=b"some ffmpeg error",
        )

    monkeypatch.setattr(audio.subprocess, "run", _raise)

    with pytest.raises(FfmpegFailed, match="some ffmpeg error"):
        normalize_to_wav(Path("in.opus"), Path("out.wav"))
