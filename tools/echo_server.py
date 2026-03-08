"""Echo server — mirrors Otto V2 core API for local testing.

Responds to /health and /text-prompt with echo responses.
The /text-prompt endpoint echoes back whatever text is sent to it.

Usage:
    python tools/echo_server.py [--port 8080] [--host 0.0.0.0]

Then point otto-voice at it:
    ./run.sh --device 1 --outputs console clipboard otto-api --compose \
        --otto-url http://localhost:8080
"""

from __future__ import annotations

import argparse
import logging
import time
from datetime import datetime, timezone

from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("echo_server")


# --- Request/Response models (mirrors kernel/api.py) ---

class ExecutionStats(BaseModel):
    provider: str
    model: str | None = None
    duration_ms: float | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None


class TextPromptRequest(BaseModel):
    prompt: str | None = None
    text: str | None = None  # otto-voice handler sends "text" instead of "prompt"
    provider: str | None = None
    space: str | None = None
    locale: str | None = None


class TextPromptResponse(BaseModel):
    response: str
    model: str | None = None
    stats: ExecutionStats | None = None
    timestamp: str


class HealthResponse(BaseModel):
    status: str
    timestamp: str
    modules: list[dict]


# --- App ---

app = FastAPI(title="Otto V2 Echo Server", version="0.0.1")


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check — reports echo server status."""
    return HealthResponse(
        status="ok",
        timestamp=datetime.now(timezone.utc).isoformat(),
        modules=[
            {"name": "echo", "version": "0.0.1", "provides": ["llm"]},
        ],
    )


@app.post("/text-prompt", response_model=TextPromptResponse)
async def text_prompt(request: TextPromptRequest):
    """Echo back the received text."""
    start = time.monotonic()

    # Accept either "prompt" or "text" field
    input_text = request.prompt or request.text or ""
    provider = request.provider or "echo"

    logger.info("Received: '%s' (provider=%s)", input_text, provider)

    response_text = input_text
    duration_ms = (time.monotonic() - start) * 1000

    return TextPromptResponse(
        response=response_text,
        model="echo-v1",
        stats=ExecutionStats(
            provider=provider,
            model="echo-v1",
            duration_ms=round(duration_ms, 2),
            input_tokens=len(input_text.split()),
            output_tokens=len(response_text.split()),
        ),
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


@app.get("/modules")
async def list_modules():
    """List loaded modules."""
    return {
        "modules": [
            {
                "name": "echo",
                "version": "0.0.1",
                "provides": ["llm"],
                "dependencies": [],
            },
        ],
    }


def main():
    parser = argparse.ArgumentParser(description="Otto V2 Echo Server")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8080, help="Bind port (default: 8080)")
    args = parser.parse_args()

    logger.info("Starting echo server on %s:%d", args.host, args.port)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
