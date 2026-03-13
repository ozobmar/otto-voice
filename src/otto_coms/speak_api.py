"""Speak API — reverse voice interface.

Exposes a FastAPI HTTP server so external callers (e.g. Claude via WebFetch)
can send text, have it spoken via TTS, and receive the user's spoken response
as transcribed text.

Usage:
    otto-coms --mode speak-api [--api-host 0.0.0.0] [--api-port 8766]

Only one concurrent speak request is processed at a time. Additional requests
block until the current one completes or times out.

No output handlers are active in this mode.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Optional

import numpy as np
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

from otto_coms.audio.capture import AudioCapture
from otto_coms.config import Config
from otto_coms.platform.audio_feedback import beep_start, beep_done
from otto_coms.processing.stt import STTEngine
from otto_coms.processing.vad import VADProcessor
from otto_coms.tts.engine import TTSEngine

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# API models
# ---------------------------------------------------------------------------

class SpeakRequest(BaseModel):
    text: str
    timeout: float = 30.0


class SpeakResponse(BaseModel):
    text: str
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Shared pipeline state
# ---------------------------------------------------------------------------

class SpeakApiState:
    """Coordinates the HTTP handler and audio pipeline loop."""

    def __init__(self) -> None:
        self.lock: asyncio.Lock = asyncio.Lock()
        self.response_future: Optional[asyncio.Future[str]] = None
        self.listening: bool = False
        self.tts_engine: Optional[TTSEngine] = None
        self.ready: asyncio.Event = asyncio.Event()  # set once pipeline is ready


_state: SpeakApiState = SpeakApiState()
_app: FastAPI = FastAPI(title="otto-coms speak-api")


# ---------------------------------------------------------------------------
# FastAPI endpoints
# ---------------------------------------------------------------------------

@_app.get("/health")
async def health() -> dict:
    return {"status": "ok", "listening": _state.listening, "ready": _state.ready.is_set()}


@_app.post("/speak", response_model=SpeakResponse)
async def speak(req: SpeakRequest) -> SpeakResponse:
    """Speak text via TTS, listen for user response, return transcription."""

    # Wait for pipeline to be initialised
    try:
        await asyncio.wait_for(_state.ready.wait(), timeout=10.0)
    except asyncio.TimeoutError:
        return SpeakResponse(text="", error="pipeline not ready")

    async with _state.lock:
        loop = asyncio.get_running_loop()

        # Open the future and start listening BEFORE TTS — this allows the
        # audio loop to capture TTS playback (self-test via Stereo Mix) or
        # any speech that overlaps with the tail of TTS playback.
        _state.response_future = loop.create_future()
        _state.listening = True

        # Speak the text
        if _state.tts_engine is not None:
            logger.info("[speak-api] Speaking: %s", req.text)
            _state.tts_engine.speak(req.text)
            # Wait for TTS to finish (mic is already open)
            await loop.run_in_executor(None, _state.tts_engine.wait)
            logger.info("[speak-api] TTS done, listening for response...")

        try:
            text = await asyncio.wait_for(
                asyncio.shield(_state.response_future),
                timeout=req.timeout,
            )
            logger.info("[speak-api] Got response: %s", text)
            return SpeakResponse(text=text)
        except asyncio.TimeoutError:
            logger.warning("[speak-api] Timeout waiting for response")
            return SpeakResponse(text="", error="timeout")
        finally:
            _state.listening = False
            _state.response_future = None


# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------

def _resolve_audio_device(config_device: int | str | None) -> int | None:
    import sounddevice as sd
    if config_device is None:
        return None
    if isinstance(config_device, str):
        try:
            info = sd.query_devices(config_device)
            return info["index"]
        except ValueError:
            logger.error("Audio device '%s' not found", config_device)
            return None
    return config_device


async def _audio_loop(
    config: Config,
    audio_queue: asyncio.Queue,
    vad: VADProcessor,
    stt: STTEngine,
) -> None:
    """Process audio chunks: VAD -> STT -> resolve response future."""
    loop = asyncio.get_running_loop()

    import sounddevice as sd
    out_cfg = config.audio.output_device or config.audio.device
    in_cfg = config.audio.input_device or config.audio.device
    out_dev = _resolve_audio_device(out_cfg)
    in_dev = _resolve_audio_device(in_cfg)
    sd.default.device = (in_dev, out_dev)

    capture = AudioCapture(config.audio, audio_queue)
    await capture.start()
    last_chunk = time.monotonic()

    try:
        while True:
            try:
                chunk = await asyncio.wait_for(audio_queue.get(), timeout=0.1)
            except asyncio.TimeoutError:
                if (time.monotonic() - last_chunk > config.audio.reconnect_timeout_s):
                    logger.warning("[speak-api] Audio timeout, reconnecting...")
                    try:
                        await capture.stop()
                    except Exception:
                        pass
                    await asyncio.sleep(2.0)
                    await capture.start()
                    last_chunk = time.monotonic()
                continue

            last_chunk = time.monotonic()

            # Skip audio while TTS is playing — unless actively listening
            # (when listening, we want to capture TTS playback for self-test)
            if not _state.listening and _state.tts_engine is not None and _state.tts_engine.is_playing:
                continue

            # Only process VAD/STT when a speak request is waiting
            if not _state.listening:
                vad.reset()
                continue

            segment = vad.process_chunk(chunk)
            if segment is None:
                continue

            # Transcribe
            loop.run_in_executor(None, beep_start)
            text = await loop.run_in_executor(
                None, stt.transcribe, segment, config.audio.sample_rate,
            )
            loop.run_in_executor(None, beep_done)

            if not text:
                continue

            # Resolve the pending future
            if (_state.listening
                    and _state.response_future is not None
                    and not _state.response_future.done()):
                _state.response_future.set_result(text)

    finally:
        try:
            await asyncio.wait_for(capture.stop(), timeout=2.0)
        except Exception:
            pass


async def run_speak_api(config: Config, host: str = "0.0.0.0", port: int = 8766) -> None:
    """Entry point for speak-api mode. Starts the pipeline and HTTP server."""

    # Force no output handlers
    config.outputs = []

    # Require TTS
    if not config.tts.enabled:
        logger.warning("[speak-api] TTS is disabled — enabling it now")
        config.tts.enabled = True

    # Initialise components
    loop = asyncio.get_running_loop()
    audio_queue: asyncio.Queue[np.ndarray] = asyncio.Queue(maxsize=2000)
    vad = VADProcessor(config.vad, sample_rate=config.audio.sample_rate)
    stt = STTEngine(config.stt)
    tts = TTSEngine(voice=config.tts.voice, rate=config.tts.rate, volume=config.tts.volume)

    logger.info("[speak-api] Loading STT model...")
    await loop.run_in_executor(None, stt.load)

    logger.info("[speak-api] Loading VAD model...")
    await loop.run_in_executor(None, vad._load_model)

    if not tts.load():
        raise RuntimeError("TTS engine failed to load")

    _state.tts_engine = tts

    # Start audio pipeline loop
    pipeline_task = asyncio.create_task(
        _audio_loop(config, audio_queue, vad, stt)
    )

    # Signal ready
    _state.ready.set()
    logger.info("[speak-api] Ready — listening on %s:%d", host, port)
    print(f"\n[SPEAK API] Ready on http://{host}:{port}/speak\n")

    # Start uvicorn server
    uv_config = uvicorn.Config(
        app=_app,
        host=host,
        port=port,
        log_level="warning",
    )
    server = uvicorn.Server(uv_config)

    try:
        await server.serve()
    finally:
        pipeline_task.cancel()
        try:
            await asyncio.wait_for(pipeline_task, timeout=3.0)
        except (asyncio.CancelledError, asyncio.TimeoutError):
            pass
        tts.stop()
        logger.info("[speak-api] Stopped.")
