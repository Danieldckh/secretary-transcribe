from __future__ import annotations

import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Literal

import openai
from fastapi import Depends, FastAPI, File, HTTPException, Query, Response, UploadFile
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from secretary_transcribe.audio import FfmpegFailed, FfmpegMissing
from secretary_transcribe.auth import require_api_key
from secretary_transcribe.config import _ensure_dotenv_loaded
from secretary_transcribe.pipeline import run_pipeline
from secretary_transcribe.translate import detect_language, translate_text

_ensure_dotenv_loaded()

ALLOWED_EXTS = {
    ".opus", ".ogg", ".m4a", ".mp3", ".mp4",
    ".mpeg", ".mpga", ".wav", ".webm", ".flac", ".aac",
}
MAX_BYTES = 200 * 1024 * 1024
MAX_MB = MAX_BYTES // (1024 * 1024)
CHUNK_SIZE = 1024 * 1024

_LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=_LOG_LEVEL, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("secretary_transcribe")
logger.setLevel(_LOG_LEVEL)

_STATIC_DIR = Path(__file__).parent / "static"
_INDEX_HTML = _STATIC_DIR / "index.html"

app = FastAPI(title="Secretary Transcribe", version="0.1.0")
app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/")
async def index() -> Response:
    html = _INDEX_HTML.read_text(encoding="utf-8")
    api_key = os.environ.get("API_KEY", "")
    html = html.replace("__SECRETARY_API_KEY__", api_key)
    return Response(content=html, media_type="text/html")


@app.post("/api/transcribe", dependencies=[Depends(require_api_key)])
async def transcribe(
    file: UploadFile = File(...),
    model: Literal["full", "mini"] = Query("full"),
    context: str | None = Query(None),
) -> dict:
    filename = file.filename or ""
    ext = Path(filename).suffix.lower()
    if not ext or ext not in ALLOWED_EXTS:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type: {ext or '(no extension)'}",
        )

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
    temp_path = Path(tmp.name)
    total = 0
    too_large = False
    try:
        try:
            while True:
                chunk = await file.read(CHUNK_SIZE)
                if not chunk:
                    break
                total += len(chunk)
                if total > MAX_BYTES:
                    too_large = True
                    break
                tmp.write(chunk)
        finally:
            tmp.close()

        if too_large:
            raise HTTPException(
                status_code=413,
                detail=f"File too large: max {MAX_BYTES} bytes ({MAX_MB} MB)",
            )

        started = time.monotonic()
        try:
            result = await run_pipeline(
                temp_path,
                cheap=(model == "mini"),
                context=context,
            )
        except FfmpegMissing as exc:
            raise HTTPException(
                status_code=500,
                detail=f"ffmpeg is not installed on the server: {exc}",
            ) from exc
        except FfmpegFailed as exc:
            raise HTTPException(
                status_code=500,
                detail=f"ffmpeg failed to process audio: {exc}",
            ) from exc
        except RuntimeError as exc:
            msg = str(exc)
            if "No speech detected" in msg:
                raise HTTPException(status_code=422, detail=msg) from exc
            raise
        except openai.APIError as exc:
            raise HTTPException(
                status_code=502,
                detail={"error": "Upstream OpenAI error", "detail": str(exc)},
            ) from exc

        elapsed = time.monotonic() - started
        logger.info(
            "transcribe filename=%s size=%d duration=%.2f model=%s total_seconds=%.2f",
            filename,
            total,
            result.duration_seconds,
            result.model,
            elapsed,
        )
        return {"filename": filename, **result.model_dump()}
    finally:
        try:
            temp_path.unlink(missing_ok=True)
        except OSError:
            pass


class TranslateRequest(BaseModel):
    text: str = Field(..., min_length=1)
    source_lang: Literal["auto", "en", "af"] = "auto"
    target_lang: Literal["en", "af"]
    model: Literal["full", "mini"] = "full"
    context: str | None = None


class TranslateResponse(BaseModel):
    source_lang: Literal["en", "af"]
    target_lang: Literal["en", "af"]
    source_text: str
    translated_text: str
    model: str


class DetectLanguageRequest(BaseModel):
    text: str = Field(..., min_length=1)


class DetectLanguageResponse(BaseModel):
    language: Literal["en", "af", "other"]
    confidence: float = Field(..., ge=0.0, le=1.0)


@app.post(
    "/api/translate",
    dependencies=[Depends(require_api_key)],
    response_model=TranslateResponse,
)
async def translate(body: TranslateRequest) -> TranslateResponse:
    cheap = body.model == "mini"
    started = time.monotonic()

    try:
        resolved_source: Literal["en", "af"]
        if body.source_lang == "auto":
            detected, _confidence, _detect_model = await detect_language(body.text)
            resolved_source = "af" if detected == "af" else "en"
        else:
            resolved_source = body.source_lang

        if resolved_source == body.target_lang:
            elapsed = time.monotonic() - started
            logger.info(
                "translate short_circuit source=%s target=%s len=%d elapsed=%.2f",
                resolved_source,
                body.target_lang,
                len(body.text),
                elapsed,
            )
            return TranslateResponse(
                source_lang=resolved_source,
                target_lang=body.target_lang,
                source_text=body.text,
                translated_text=body.text,
                model="passthrough",
            )

        translated_text, model_name = await translate_text(
            body.text,
            source_lang=resolved_source,
            target_lang=body.target_lang,
            cheap=cheap,
            context=body.context,
        )
    except openai.APIError as exc:
        raise HTTPException(
            status_code=502,
            detail={"error": "Upstream OpenAI error", "detail": str(exc)},
        ) from exc
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001  unexpected path -> 500
        logger.exception("translate unexpected_error")
        raise HTTPException(
            status_code=500,
            detail={"error": "Unexpected server error", "detail": str(exc)},
        ) from exc

    elapsed = time.monotonic() - started
    logger.info(
        "translate source=%s target=%s len=%d model=%s elapsed=%.2f",
        resolved_source,
        body.target_lang,
        len(body.text),
        model_name,
        elapsed,
    )
    return TranslateResponse(
        source_lang=resolved_source,
        target_lang=body.target_lang,
        source_text=body.text,
        translated_text=translated_text,
        model=model_name,
    )


@app.post(
    "/api/detect-language",
    dependencies=[Depends(require_api_key)],
    response_model=DetectLanguageResponse,
)
async def detect_language_route(body: DetectLanguageRequest) -> DetectLanguageResponse:
    started = time.monotonic()
    try:
        language, confidence, model_name = await detect_language(body.text)
    except openai.APIError as exc:
        raise HTTPException(
            status_code=502,
            detail={"error": "Upstream OpenAI error", "detail": str(exc)},
        ) from exc
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001  unexpected path -> 500
        logger.exception("detect_language unexpected_error")
        raise HTTPException(
            status_code=500,
            detail={"error": "Unexpected server error", "detail": str(exc)},
        ) from exc

    elapsed = time.monotonic() - started
    logger.info(
        "detect_language len=%d language=%s confidence=%.2f model=%s elapsed=%.2f",
        len(body.text),
        language,
        confidence,
        model_name,
        elapsed,
    )
    return DetectLanguageResponse(language=language, confidence=confidence)
