from __future__ import annotations

import io
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient


def _audio_file(name: str = "voicenote.m4a", size: int = 64) -> tuple[str, io.BytesIO, str]:
    return (name, io.BytesIO(b"\x00" * size), "audio/mp4")


def test_health_no_auth(client: TestClient) -> None:
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json() == {"status": "ok"}


def test_index_returns_html(client: TestClient) -> None:
    res = client.get("/")
    assert res.status_code == 200
    assert res.headers["content-type"].startswith("text/html")


def test_transcribe_missing_key_401(client: TestClient) -> None:
    res = client.post("/api/transcribe", files={"file": _audio_file()})
    assert res.status_code == 401


def test_transcribe_wrong_key_401(client: TestClient) -> None:
    res = client.post(
        "/api/transcribe",
        headers={"X-API-Key": "wrong"},
        files={"file": _audio_file()},
    )
    assert res.status_code == 401


def test_transcribe_happy_path(client: TestClient, api_key: str) -> None:
    res = client.post(
        "/api/transcribe",
        headers={"X-API-Key": api_key},
        files={"file": _audio_file("note.m4a")},
    )
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["filename"] == "note.m4a"
    assert body["afrikaans"] == "Hallo wêreld"
    assert body["english"] == "Hello world"
    assert body["model"] == "gpt-4o"
    assert body["duration_seconds"] == 3.21


def test_transcribe_unsupported_extension(client: TestClient, api_key: str) -> None:
    res = client.post(
        "/api/transcribe",
        headers={"X-API-Key": api_key},
        files={"file": ("notes.txt", io.BytesIO(b"hello"), "text/plain")},
    )
    assert res.status_code == 415


def test_transcribe_too_large(client: TestClient, api_key: str) -> None:
    big = b"\x00" * (200 * 1024 * 1024 + 1)
    res = client.post(
        "/api/transcribe",
        headers={"X-API-Key": api_key},
        files={"file": ("big.m4a", io.BytesIO(big), "audio/mp4")},
    )
    assert res.status_code == 413


def test_transcribe_25mb_now_accepted(client: TestClient, api_key: str) -> None:
    """Files between the old 25 MB cap and the new 200 MB cap should now be accepted."""
    payload = b"\x00" * (30 * 1024 * 1024)
    res = client.post(
        "/api/transcribe",
        headers={"X-API-Key": api_key},
        files={"file": ("medium.m4a", io.BytesIO(payload), "audio/mp4")},
    )
    assert res.status_code == 200, res.text


def test_transcribe_no_speech(client: TestClient, api_key: str, mock_pipeline) -> None:
    mock_pipeline.side_effect = RuntimeError("No speech detected")
    res = client.post(
        "/api/transcribe",
        headers={"X-API-Key": api_key},
        files={"file": _audio_file()},
    )
    assert res.status_code == 422
    assert "No speech detected" in res.json()["detail"]


def test_transcribe_cheap_query_param(client: TestClient, api_key: str, mock_pipeline) -> None:
    res = client.post(
        "/api/transcribe?model=mini",
        headers={"X-API-Key": api_key},
        files={"file": _audio_file()},
    )
    assert res.status_code == 200
    mock_pipeline.assert_awaited_once()
    kwargs = mock_pipeline.await_args.kwargs
    assert kwargs["cheap"] is True


def test_transcribe_context_query_param(client: TestClient, api_key: str, mock_pipeline) -> None:
    res = client.post(
        "/api/transcribe?context=citrus",
        headers={"X-API-Key": api_key},
        files={"file": _audio_file()},
    )
    assert res.status_code == 200
    mock_pipeline.assert_awaited_once()
    kwargs = mock_pipeline.await_args.kwargs
    assert kwargs["context"] == "citrus"


@pytest.fixture
def mock_translate_text(monkeypatch: pytest.MonkeyPatch) -> AsyncMock:
    from secretary_transcribe import app as app_module

    mock = AsyncMock(return_value=("translated", "gpt-4o"))
    monkeypatch.setattr(app_module, "translate_text", mock)
    return mock


@pytest.fixture
def mock_detect_language(monkeypatch: pytest.MonkeyPatch) -> AsyncMock:
    from secretary_transcribe import app as app_module

    mock = AsyncMock(return_value=("af", 0.97, "gpt-4o-mini"))
    monkeypatch.setattr(app_module, "detect_language", mock)
    return mock


def test_translate_missing_key_401(client: TestClient) -> None:
    res = client.post("/api/translate", json={"text": "hi", "target_lang": "af"})
    assert res.status_code == 401


def test_translate_happy_path(
    client: TestClient,
    api_key: str,
    mock_translate_text: AsyncMock,
    mock_detect_language: AsyncMock,
) -> None:
    res = client.post(
        "/api/translate",
        headers={"X-API-Key": api_key},
        json={"text": "Hello world", "source_lang": "en", "target_lang": "af"},
    )
    assert res.status_code == 200, res.text
    body = res.json()
    assert body == {
        "source_lang": "en",
        "target_lang": "af",
        "source_text": "Hello world",
        "translated_text": "translated",
        "model": "gpt-4o",
    }
    mock_detect_language.assert_not_awaited()
    mock_translate_text.assert_awaited_once()


def test_translate_auto_detect_routes_through_detector(
    client: TestClient,
    api_key: str,
    mock_translate_text: AsyncMock,
    mock_detect_language: AsyncMock,
) -> None:
    res = client.post(
        "/api/translate",
        headers={"X-API-Key": api_key},
        json={"text": "Hallo", "source_lang": "auto", "target_lang": "en"},
    )
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["source_lang"] == "af"
    mock_detect_language.assert_awaited_once()
    mock_translate_text.assert_awaited_once()


def test_translate_short_circuit_when_source_equals_target(
    client: TestClient,
    api_key: str,
    mock_translate_text: AsyncMock,
    mock_detect_language: AsyncMock,
) -> None:
    res = client.post(
        "/api/translate",
        headers={"X-API-Key": api_key},
        json={"text": "no change", "source_lang": "en", "target_lang": "en"},
    )
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["translated_text"] == body["source_text"] == "no change"
    assert body["model"] == "passthrough"
    mock_translate_text.assert_not_awaited()
    mock_detect_language.assert_not_awaited()


def test_translate_validation_error_422(client: TestClient, api_key: str) -> None:
    res = client.post(
        "/api/translate",
        headers={"X-API-Key": api_key},
        json={"text": "", "target_lang": "af"},
    )
    assert res.status_code == 422


def test_detect_language_missing_key_401(client: TestClient) -> None:
    res = client.post("/api/detect-language", json={"text": "Hallo"})
    assert res.status_code == 401


def test_detect_language_happy_path(
    client: TestClient,
    api_key: str,
    mock_detect_language: AsyncMock,
) -> None:
    res = client.post(
        "/api/detect-language",
        headers={"X-API-Key": api_key},
        json={"text": "Hallo, hoe gaan dit?"},
    )
    assert res.status_code == 200, res.text
    body = res.json()
    assert body == {"language": "af", "confidence": 0.97}
    mock_detect_language.assert_awaited_once()
