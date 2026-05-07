from __future__ import annotations

import subprocess
from pathlib import Path


class FfmpegMissing(RuntimeError):
    pass


class FfmpegFailed(RuntimeError):
    pass


def normalize_to_wav(input_path: Path, output_path: Path) -> None:
    """Convert input audio to 16 kHz mono PCM WAV using ffmpeg."""
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_path),
        "-ac",
        "1",
        "-ar",
        "16000",
        "-f",
        "wav",
        str(output_path),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except FileNotFoundError as exc:
        raise FfmpegMissing(
            "ffmpeg executable not found on PATH; install ffmpeg or set the binary location"
        ) from exc
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or b"").decode("utf-8", errors="replace").strip()
        raise FfmpegFailed(
            f"ffmpeg failed (exit {exc.returncode}) for {input_path}: {stderr}"
        ) from exc
