from __future__ import annotations

import io

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
    big = b"\x00" * (25 * 1024 * 1024 + 1)
    res = client.post(
        "/api/transcribe",
        headers={"X-API-Key": api_key},
        files={"file": ("big.m4a", io.BytesIO(big), "audio/mp4")},
    )
    assert res.status_code == 413


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
