from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from secretary_transcribe import audio
from secretary_transcribe.audio import (
    FfmpegFailed,
    FfmpegMissing,
    normalize_to_opus,
    normalize_to_wav,
)
from secretary_transcribe.pipeline import (
    CHUNK_DURATION_SECONDS,
    WHISPER_SAFE_BYTES,
    PipelineResult,
    run_pipeline,
)


@pytest.fixture(autouse=True)
def _set_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test")


def _fake_whisper_response(text: str = "Hallo wereld", duration: float = 4.2) -> SimpleNamespace:
    return SimpleNamespace(text=text, duration=duration, segments=[])


def _stub_audio(monkeypatch: pytest.MonkeyPatch, *, duration: float, size_bytes: int = 1024) -> None:
    """Stub normalize_to_opus to write a fake file of given size, and pin reported duration."""
    def fake_normalize(input_path: Path, output_path: Path, **_kwargs: object) -> None:
        output_path.write_bytes(b"\x00" * size_bytes)

    monkeypatch.setattr(
        "secretary_transcribe.audio.normalize_to_opus", fake_normalize
    )
    monkeypatch.setattr(
        "secretary_transcribe.audio.get_duration_seconds", lambda *_a, **_k: duration
    )


@pytest.mark.asyncio
async def test_run_pipeline_happy_path(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_audio(monkeypatch, duration=4.2)
    transcribe_mock = AsyncMock(return_value=_fake_whisper_response("Hallo wereld", 4.2))
    monkeypatch.setattr(
        "secretary_transcribe.transcribe.transcribe_afrikaans", transcribe_mock
    )
    fix_mock = AsyncMock(return_value=("Hallo wereld (simpel)", "gpt-4o"))
    monkeypatch.setattr(
        "secretary_transcribe.translate.fix_afrikaans_construction", fix_mock
    )
    translate_mock = AsyncMock(return_value=("Hello world", "gpt-4o"))
    monkeypatch.setattr(
        "secretary_transcribe.translate.translate_to_english", translate_mock
    )

    result = await run_pipeline(Path("fake.opus"))

    assert isinstance(result, PipelineResult)
    assert result.afrikaans == "Hallo wereld (simpel)"
    assert result.english == "Hello world"
    assert result.duration_seconds == pytest.approx(4.2)
    assert result.model == "gpt-4o"
    transcribe_mock.assert_awaited_once()
    fix_mock.assert_awaited_once()
    translate_mock.assert_awaited_once()
    assert translate_mock.await_args.args[0] == "Hallo wereld (simpel)"


@pytest.mark.asyncio
async def test_run_pipeline_cheap_uses_mini(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_audio(monkeypatch, duration=1.0)
    monkeypatch.setattr(
        "secretary_transcribe.transcribe.transcribe_afrikaans",
        AsyncMock(return_value=_fake_whisper_response("Hallo", 1.0)),
    )
    fix_mock = AsyncMock(return_value=("Hallo", "gpt-4o-mini"))
    monkeypatch.setattr(
        "secretary_transcribe.translate.fix_afrikaans_construction", fix_mock
    )
    translate_mock = AsyncMock(return_value=("Hello", "gpt-4o-mini"))
    monkeypatch.setattr(
        "secretary_transcribe.translate.translate_to_english", translate_mock
    )

    result = await run_pipeline(Path("fake.opus"), cheap=True)

    assert result.model == "gpt-4o-mini"
    _, fix_kwargs = fix_mock.await_args
    assert fix_kwargs.get("cheap") is True
    _, tr_kwargs = translate_mock.await_args
    assert tr_kwargs.get("cheap") is True


@pytest.mark.asyncio
async def test_run_pipeline_custom_context(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_audio(monkeypatch, duration=2.0)
    monkeypatch.setattr(
        "secretary_transcribe.transcribe.transcribe_afrikaans",
        AsyncMock(return_value=_fake_whisper_response("Sitrus uitvoer", 2.0)),
    )
    fix_mock = AsyncMock(return_value=("Sitrus uitvoer (simpel)", "gpt-4o"))
    monkeypatch.setattr(
        "secretary_transcribe.translate.fix_afrikaans_construction", fix_mock
    )
    translate_mock = AsyncMock(return_value=("Citrus exports", "gpt-4o"))
    monkeypatch.setattr(
        "secretary_transcribe.translate.translate_to_english", translate_mock
    )

    custom = "discussion about citrus exports"
    result = await run_pipeline(Path("fake.opus"), context=custom)

    assert result.english == "Citrus exports"
    assert result.afrikaans == "Sitrus uitvoer (simpel)"
    _, fix_kwargs = fix_mock.await_args
    assert fix_kwargs.get("context") == custom
    _, tr_kwargs = translate_mock.await_args
    assert tr_kwargs.get("context") == custom


@pytest.mark.asyncio
async def test_run_pipeline_empty_transcription_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_audio(monkeypatch, duration=0.5)
    monkeypatch.setattr(
        "secretary_transcribe.transcribe.transcribe_afrikaans",
        AsyncMock(return_value=_fake_whisper_response("   \n\t", 0.5)),
    )
    fix_mock = AsyncMock(return_value=("nooit", "gpt-4o"))
    monkeypatch.setattr(
        "secretary_transcribe.translate.fix_afrikaans_construction", fix_mock
    )
    translate_mock = AsyncMock(return_value=("should not be called", "gpt-4o"))
    monkeypatch.setattr(
        "secretary_transcribe.translate.translate_to_english", translate_mock
    )

    with pytest.raises(RuntimeError, match="No speech"):
        await run_pipeline(Path("fake.opus"))

    fix_mock.assert_not_awaited()
    translate_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_run_pipeline_chunks_long_audio(monkeypatch: pytest.MonkeyPatch) -> None:
    """Audio longer than CHUNK_DURATION_SECONDS triggers split_into_chunks and multiple transcription calls."""
    long_duration = float(CHUNK_DURATION_SECONDS * 3 + 100)
    _stub_audio(monkeypatch, duration=long_duration, size_bytes=1024)

    def fake_split(audio_path: Path, output_dir: Path, **_kwargs: object) -> list[Path]:
        paths = []
        for i in range(3):
            p = output_dir / f"chunk_{i}.opus"
            p.write_bytes(b"\x00" * 256)
            paths.append(p)
        return paths

    monkeypatch.setattr("secretary_transcribe.audio.split_into_chunks", fake_split)

    transcripts = ["Een ", "twee ", "drie"]
    transcribe_mock = AsyncMock(
        side_effect=[_fake_whisper_response(t, 200.0) for t in transcripts]
    )
    monkeypatch.setattr(
        "secretary_transcribe.transcribe.transcribe_afrikaans", transcribe_mock
    )
    fix_mock = AsyncMock(return_value=("Een twee drie", "gpt-4o"))
    monkeypatch.setattr(
        "secretary_transcribe.translate.fix_afrikaans_construction", fix_mock
    )
    translate_mock = AsyncMock(return_value=("One two three", "gpt-4o"))
    monkeypatch.setattr(
        "secretary_transcribe.translate.translate_to_english", translate_mock
    )

    result = await run_pipeline(Path("fake.opus"))

    assert transcribe_mock.await_count == 3
    assert fix_mock.await_args.args[0] == "Een twee drie"
    assert result.english == "One two three"
    assert result.duration_seconds == pytest.approx(long_duration)


@pytest.mark.asyncio
async def test_run_pipeline_chunks_oversize_short_audio(monkeypatch: pytest.MonkeyPatch) -> None:
    """File larger than WHISPER_SAFE_BYTES triggers chunking even when duration is short."""
    _stub_audio(monkeypatch, duration=60.0, size_bytes=WHISPER_SAFE_BYTES + 1)

    def fake_split(audio_path: Path, output_dir: Path, **_kwargs: object) -> list[Path]:
        p = output_dir / "chunk_0.opus"
        p.write_bytes(b"\x00" * 256)
        return [p]

    monkeypatch.setattr("secretary_transcribe.audio.split_into_chunks", fake_split)
    transcribe_mock = AsyncMock(return_value=_fake_whisper_response("Klaar", 60.0))
    monkeypatch.setattr(
        "secretary_transcribe.transcribe.transcribe_afrikaans", transcribe_mock
    )
    monkeypatch.setattr(
        "secretary_transcribe.translate.fix_afrikaans_construction",
        AsyncMock(return_value=("Klaar", "gpt-4o")),
    )
    monkeypatch.setattr(
        "secretary_transcribe.translate.translate_to_english",
        AsyncMock(return_value=("Done", "gpt-4o")),
    )

    result = await run_pipeline(Path("fake.opus"))

    transcribe_mock.assert_awaited_once()
    assert result.english == "Done"


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


def test_normalize_to_opus_invokes_libopus(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: list[list[str]] = []

    def fake_run(cmd: list[str], **_kwargs: object) -> subprocess.CompletedProcess[bytes]:
        captured.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout=b"", stderr=b"")

    monkeypatch.setattr(audio.subprocess, "run", fake_run)
    normalize_to_opus(Path("in.m4a"), Path("out.opus"))

    assert captured, "ffmpeg should have been invoked"
    cmd = captured[0]
    assert "libopus" in cmd
    assert "16000" in cmd


def test_get_duration_seconds_parses_ffprobe_output(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(cmd: list[str], **_kwargs: object) -> subprocess.CompletedProcess[bytes]:
        return subprocess.CompletedProcess(cmd, 0, stdout=b"123.45\n", stderr=b"")

    monkeypatch.setattr(audio.subprocess, "run", fake_run)
    assert audio.get_duration_seconds(Path("anything.opus")) == pytest.approx(123.45)


def test_split_into_chunks_creates_overlapping_segments(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        "secretary_transcribe.audio.get_duration_seconds", lambda *_a, **_k: 1300.0
    )

    captured_starts: list[float] = []

    def fake_run(cmd: list[str], **_kwargs: object) -> subprocess.CompletedProcess[bytes]:
        if "-ss" in cmd:
            captured_starts.append(float(cmd[cmd.index("-ss") + 1]))
        for token in cmd:
            if token.endswith(".opus") and "chunk_" in token:
                Path(token).write_bytes(b"\x00")
        return subprocess.CompletedProcess(cmd, 0, stdout=b"", stderr=b"")

    monkeypatch.setattr(audio.subprocess, "run", fake_run)

    chunks = audio.split_into_chunks(
        Path("source.opus"),
        tmp_path,
        chunk_seconds=600,
        overlap_seconds=5,
    )

    assert len(chunks) >= 2
    assert captured_starts[0] == 0.0
    assert captured_starts[1] == pytest.approx(595.0)
