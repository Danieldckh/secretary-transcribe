# secretary-transcribe

Hosted Afrikaans to English transcription service for WhatsApp voice notes.

## What it does

Accepts a WhatsApp voice note (`.opus`, `.m4a`, `.ogg`, `.wav`, `.mp3`), transcribes the Afrikaans speech, and returns an English translation. The pipeline is tuned by default for the South African agricultural domain — vocabulary, place names, and idioms common in farming contexts are biased into the prompt to improve accuracy on field recordings.

## API

Three endpoints are exposed:

### `GET /`

Static landing page with a minimal upload UI.

### `GET /health`

Liveness probe. Returns `200 OK` with `{"status": "ok"}` when the service is up.

```bash
curl https://transcribe.proagrihub.com/health
```

### `POST /api/transcribe`

Authenticated. Accepts a multipart upload with the audio file under field `file`.

Request:

```bash
curl -X POST https://transcribe.proagrihub.com/api/transcribe \
  -H "X-API-Key: $API_KEY" \
  -F "file=@voicenote.opus"
```

Response:

```json
{
  "afrikaans": "Goeiemôre, ek wil net laat weet die mielies is reg vir oes.",
  "english": "Good morning, I just want to let you know the maize is ready for harvest.",
  "duration_seconds": 4.2
}
```

## Environment variables

| Name             | Required | Description                                                          |
| ---------------- | -------- | -------------------------------------------------------------------- |
| `OPENAI_API_KEY` | yes      | OpenAI API key used for Whisper transcription and translation calls. |
| `API_KEY`        | yes      | Shared secret callers send in the `X-API-Key` header.                |
| `LOG_LEVEL`      | no       | Python log level. Defaults to `INFO`.                                |

## Run locally

Requires Python 3.12+ and `ffmpeg` on PATH.

Windows:

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
```

Unix:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Create a `.env` file in the project root with:

```
OPENAI_API_KEY=sk-...
API_KEY=some-shared-secret
```

Then run:

```bash
uvicorn secretary_transcribe.app:app --reload
```

Open http://localhost:8000.

## Run tests

```bash
pytest
```

## Build and run with Docker

```bash
docker build -t secretary-transcribe .
docker run --rm -p 8000:8000 \
  -e OPENAI_API_KEY=sk-... \
  -e API_KEY=some-shared-secret \
  secretary-transcribe
```

## Deploy to Coolify

The service is auto-deployed by Coolify (https://coolify.proagrihub.com) from the `main` branch of the GitHub repository. Environment variables are configured in the Coolify UI and injected at runtime — no secrets live in the image. The container exposes port 8000; Coolify handles TLS termination and reverse proxying.

## License

Internal — not licensed for redistribution.
