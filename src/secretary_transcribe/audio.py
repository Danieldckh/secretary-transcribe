from __future__ import annotations

import math
import subprocess
from pathlib import Path


class FfmpegMissing(RuntimeError):
    pass


class FfmpegFailed(RuntimeError):
    pass


def _run(cmd: list[str]) -> subprocess.CompletedProcess[bytes]:
    try:
        return subprocess.run(cmd, check=True, capture_output=True)
    except FileNotFoundError as exc:
        raise FfmpegMissing(
            f"{cmd[0]} executable not found on PATH; install ffmpeg or set the binary location"
        ) from exc
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or b"").decode("utf-8", errors="replace").strip()
        raise FfmpegFailed(
            f"{cmd[0]} failed (exit {exc.returncode}): {stderr}"
        ) from exc


def normalize_to_wav(input_path: Path, output_path: Path) -> None:
    """Convert input audio to 16 kHz mono PCM WAV using ffmpeg."""
    _run([
        "ffmpeg",
        "-y",
        "-i", str(input_path),
        "-ac", "1",
        "-ar", "16000",
        "-f", "wav",
        str(output_path),
    ])


def normalize_to_opus(input_path: Path, output_path: Path, *, bitrate: str = "24k") -> None:
    """Convert input audio to 16 kHz mono opus using ffmpeg.

    Opus at 24 kbps is ~3 KB/sec — about 10x smaller than 16 kHz PCM WAV —
    with no measurable accuracy loss for Whisper.
    """
    _run([
        "ffmpeg",
        "-y",
        "-i", str(input_path),
        "-ac", "1",
        "-ar", "16000",
        "-c:a", "libopus",
        "-b:a", bitrate,
        "-f", "ogg",
        str(output_path),
    ])


def get_duration_seconds(audio_path: Path) -> float:
    """Return the duration of an audio file in seconds using ffprobe."""
    result = _run([
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "csv=p=0",
        str(audio_path),
    ])
    raw = result.stdout.decode("utf-8", errors="replace").strip()
    if not raw:
        raise FfmpegFailed(f"ffprobe returned empty duration for {audio_path}")
    try:
        return float(raw)
    except ValueError as exc:
        raise FfmpegFailed(f"ffprobe returned non-numeric duration {raw!r}") from exc


def split_into_chunks(
    audio_path: Path,
    output_dir: Path,
    *,
    chunk_seconds: int,
    overlap_seconds: int = 0,
    bitrate: str = "24k",
) -> list[Path]:
    """Split audio into overlapping opus chunks of approximately chunk_seconds each.

    Each chunk overlaps with the next by overlap_seconds. Re-encodes per chunk to
    avoid opus container offset issues that affect stream-copy slicing.

    Returns the ordered list of chunk paths written into output_dir.
    """
    if chunk_seconds <= 0:
        raise ValueError("chunk_seconds must be positive")
    if overlap_seconds < 0 or overlap_seconds >= chunk_seconds:
        raise ValueError("overlap_seconds must be in [0, chunk_seconds)")

    duration = get_duration_seconds(audio_path)
    stride = chunk_seconds - overlap_seconds
    chunk_count = max(1, math.ceil((duration - overlap_seconds) / stride)) if duration > chunk_seconds else 1

    chunks: list[Path] = []
    for i in range(chunk_count):
        start = i * stride
        if start >= duration:
            break
        chunk_path = output_dir / f"chunk_{i:04d}.ogg"
        _run([
            "ffmpeg",
            "-y",
            "-ss", f"{start}",
            "-i", str(audio_path),
            "-t", f"{chunk_seconds}",
            "-ac", "1",
            "-ar", "16000",
            "-c:a", "libopus",
            "-b:a", bitrate,
            "-f", "ogg",
            str(chunk_path),
        ])
        chunks.append(chunk_path)
    return chunks
