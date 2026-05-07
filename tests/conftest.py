from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def api_key(monkeypatch: pytest.MonkeyPatch) -> str:
    key = "test-api-key-12345"
    monkeypatch.setenv("API_KEY", key)
    return key


@pytest.fixture
def mock_pipeline(monkeypatch: pytest.MonkeyPatch):
    from secretary_transcribe import app as app_module
    from secretary_transcribe.pipeline import PipelineResult

    result = PipelineResult(
        duration_seconds=3.21,
        afrikaans="Hallo wêreld",
        english="Hello world",
        model="gpt-4o",
    )
    mock = AsyncMock(return_value=result)
    monkeypatch.setattr(app_module, "run_pipeline", mock)
    return mock


@pytest.fixture
def client(api_key: str, mock_pipeline) -> TestClient:
    from secretary_transcribe.app import app

    return TestClient(app)
