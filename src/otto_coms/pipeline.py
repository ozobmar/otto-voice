"""Async pipeline orchestrator — audio -> VAD -> STT -> outputs.

Supports continuous and wake word listening modes.
"""

from __future__ import annotations

import asyncio
import logging
import time

import numpy as np

from otto_coms.audio.capture import AudioCapture
from otto_coms.buffer.compose import ComposeBuffer
from otto_coms.commands.voice_commands import check_voice_command
from otto_coms.commands.hotkeys import HotkeyManager
from otto_coms.config import Config
from otto_coms.handlers import create_outputs
from otto_coms.llm import create_llm_client
from otto_coms.platform.audio_feedback import beep_start, beep_done, beep_sent, beep_wake_word
from otto_coms.processing.stt import STTEngine
from otto_coms.processing.vad import VADProcessor
from otto_coms.processing.wake_word import WakeWordDetector
from otto_coms.tts.engine import TTSEngine

logger = logging.getLogger(__name__)


class PipelineState:
    """Mutable state shared across the pipeline."""

    def __init__(self) -> None:
        self.paused: bool = False
        self.listening_mode: str = "continuous"  # "continuous" or "wake_word"
        self.wake_active: bool = False  # wake word detected, actively listening
        self.wake_last_speech: float = 0.0  # monotonic time of last speech in wake mode
        self.tts_barge_in: bool = False  # set when barge-in interrupts TTS


async def run_pipeline(config: Config) -> None:
    """Main pipeline: mic -> VAD -> STT -> outputs."""

    state = PipelineState()
    state.listening_mode = config.listening.mode

    # Set default audio device for all sounddevice calls (capture, TTS, beeps)
    if config.audio.device is not None:
        import sounddevice as sd
        sd.default.device = (config.audio.device, config.audio.device)
        logger.info("Audio device set to %s (input + output)", config.audio.device)

    # Initialise components
    audio_queue: asyncio.Queue[np.ndarray] = asyncio.Queue(maxsize=500)
    capture = AudioCapture(config.audio, audio_queue)
    vad = VADProcessor(config.vad, sample_rate=config.audio.sample_rate)
    stt = STTEngine(config.stt)
    outputs = create_outputs(config)

    # Wake word detector — loaded regardless of mode so it can serve as
    # pause/resume trigger. Lightweight ONNX, negligible overhead.
    wake_detector: WakeWordDetector | None = None
    _ww = WakeWordDetector(config.listening.wake_word)
    if _ww.load():
        wake_detector = _ww
        logger.info("Wake word detector available (pause/resume + wake word mode)")
    else:
        if state.listening_mode == "wake_word":
            logger.warning("Wake word unavailable, falling back to continuous mode")
            state.listening_mode = "continuous"

    # TTS engine — speaks API responses locally
    tts_engine: TTSEngine | None = None
    if config.tts.enabled:
        _tts = TTSEngine(voice=config.tts.voice, rate=config.tts.rate, volume=config.tts.volume)
        if _tts.load():
            tts_engine = _tts

            # Connect TTS to handlers with response callbacks
            from otto_coms.handlers.otto_api import OttoApiOutput
            from otto_coms.handlers.cc_direct import CcDirectOutput
            for handler in outputs:
                if isinstance(handler, (OttoApiOutput, CcDirectOutput)):
                    def _on_response(data: dict, _tts=tts_engine) -> None:
                        response_text = data.get("response", "")
                        if response_text and _tts is not None:
                            _tts.speak(response_text)  # non-blocking, queued

                    handler.set_response_callback(_on_response)
                    handler_name = type(handler).__name__
                    logger.info("TTS wired to %s responses", handler_name)

    # Compose mode setup
    compose_mode = config.compose.enabled
    buffer: ComposeBuffer | None = None
    if compose_mode:
        auto_send_delay = config.compose.auto_send_delay_ms / 1000.0
        buffer = ComposeBuffer(auto_send_delay=auto_send_delay)
        llm_client = create_llm_client(config.llm)
        buffer.configure(
            llm=llm_client,
            outputs=outputs,
            llm_cleanup=config.compose.llm_cleanup,
        )
        buffer._on_sent = beep_sent
        logger.info(
            "Compose mode ON (auto-send=%.1fs, llm_cleanup=%s, llm=%s)",
            auto_send_delay,
            config.compose.llm_cleanup,
            "enabled" if llm_client else "disabled",
        )

    def toggle_listening() -> None:
        state.paused = not state.paused
        label = "PAUSED" if state.paused else "LISTENING"
        logger.info("Listening %s (Shift+L)", label)
        print(f"[{label}]")

    def _handle_mode_switch(mode_action: str) -> None:
        """Handle mode switching from voice commands."""
        if mode_action == "listen_continuous":
            state.listening_mode = "continuous"
            state.wake_active = False
            print("[MODE] Switched to continuous listening")
        elif mode_action == "listen_wake_word":
            if wake_detector is not None and wake_detector.available:
                state.listening_mode = "wake_word"
                state.wake_active = False
                vad.reset()
                print("[MODE] Switched to wake word listening")
            else:
                print("[MODE] Wake word not available")
        elif mode_action == "pause":
            state.paused = True
            print("[PAUSED]")
        elif mode_action == "resume":
            state.paused = False
            print("[LISTENING]")
        elif mode_action == "tx_sync":
            print("[MODE] Switched to sync transmission")
        elif mode_action == "tx_async":
            print("[MODE] Switched to async transmission")
        elif mode_action == "show_commands":
            print_commands(compose_mode)

    # Set up hotkeys
    hotkeys = HotkeyManager()
    hotkeys.bind({"shift", "l"}, toggle_listening, "Toggle listening")
    hotkeys.start()

    # Load STT model
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, stt.load)

    # Start outputs
    for handler in outputs:
        await handler.start()

    # Start audio capture
    await capture.start()
    logger.info("Pipeline running (%s mode). Ctrl+C to stop.", state.listening_mode)
    logger.info("Shift+L to pause/resume listening")

    print_commands(compose_mode)

    try:
        while True:
            try:
                chunk = await asyncio.wait_for(audio_queue.get(), timeout=0.1)
            except asyncio.TimeoutError:
                # Check wake word timeout
                if (state.listening_mode == "wake_word"
                        and state.wake_active
                        and time.monotonic() - state.wake_last_speech
                        > config.listening.wake_word.timeout_seconds):
                    state.wake_active = False
                    vad.reset()
                    logger.info("Wake word timeout — returning to listening")
                    print("[WAKE] Timeout — listening for wake word...")
                continue

            # Barge-in: when TTS is playing, detect speech and interrupt
            if tts_engine is not None and tts_engine.is_playing:
                # Wake word can still interrupt during TTS
                if wake_detector is not None and wake_detector.detect(chunk):
                    tts_engine.interrupt()
                    state.tts_barge_in = True
                    if config.listening.wake_word.feedback_beep:
                        loop.run_in_executor(None, beep_wake_word)
                    logger.info("Barge-in via wake word — TTS interrupted")
                    print("[BARGE-IN] Wake word — TTS stopped")
                    if state.listening_mode == "wake_word":
                        state.wake_active = True
                        state.wake_last_speech = time.monotonic()
                    continue

                # VAD barge-in disabled — TTS audio leaks through the mic and
                # triggers false barge-ins. Only wake word can interrupt TTS.
                # Discard audio during TTS playback to avoid processing echo.
                continue

            # Clear barge-in flag once TTS is no longer playing
            if state.tts_barge_in:
                state.tts_barge_in = False
                vad.reset()  # clean slate after barge-in

            # When paused: wake word detector stays active as resume trigger.
            # ONNX wake word is lightweight (~single CPU core for 15-20 models).
            if state.paused:
                if wake_detector is not None and wake_detector.detect(chunk):
                    state.paused = False
                    if config.listening.wake_word.feedback_beep:
                        loop.run_in_executor(None, beep_wake_word)
                    logger.info("Wake word resume from paused state")
                    print("[WAKE] Detected — resuming...")
                    # In wake word mode, also set wake_active
                    if state.listening_mode == "wake_word":
                        state.wake_active = True
                        state.wake_last_speech = time.monotonic()
                continue

            # Wake word mode: check for wake word first
            if state.listening_mode == "wake_word" and not state.wake_active:
                if wake_detector is not None and wake_detector.detect(chunk):
                    state.wake_active = True
                    state.wake_last_speech = time.monotonic()
                    if config.listening.wake_word.feedback_beep:
                        loop.run_in_executor(None, beep_wake_word)
                    print("[WAKE] Detected — listening...")
                continue

            # VAD processing
            segment = vad.process_chunk(chunk)
            if segment is None:
                # Update last speech time in wake mode
                if (state.listening_mode == "wake_word"
                        and state.wake_active
                        and vad.state.value == "speaking"):
                    state.wake_last_speech = time.monotonic()
                continue

            # Reset wake timer on speech
            if state.listening_mode == "wake_word" and state.wake_active:
                state.wake_last_speech = time.monotonic()

            logger.debug("Speech segment: %.1fs", len(segment) / config.audio.sample_rate)

            # Transcribe
            loop.run_in_executor(None, beep_start)
            text = await loop.run_in_executor(
                None, stt.transcribe, segment, config.audio.sample_rate,
            )
            if not text:
                continue

            loop.run_in_executor(None, beep_done)

            # Check voice commands
            result = check_voice_command(text, buffer)

            if result.mode_switch is not None:
                _handle_mode_switch(result.mode_switch)
                continue

            if result.handled:
                continue

            text = result.text
            if text is None:
                continue

            if compose_mode and buffer is not None:
                buffer.add(text)
            else:
                metadata = {
                    "duration": len(segment) / config.audio.sample_rate,
                    "language": config.stt.language,
                }
                for handler in outputs:
                    try:
                        await handler.emit(text, metadata)
                    except Exception as e:
                        logger.error("Output error (%s): %s", type(handler).__name__, e)
                loop.run_in_executor(None, beep_sent)

    finally:
        print("\nStopping...")
        if buffer is not None:
            buffer._cancel_auto_send()
        hotkeys.stop()
        if tts_engine is not None:
            tts_engine.stop()
        try:
            await asyncio.wait_for(capture.stop(), timeout=2.0)
        except (asyncio.TimeoutError, Exception):
            logger.warning("Audio capture stop timed out")
        for handler in outputs:
            try:
                await asyncio.wait_for(handler.stop(), timeout=2.0)
            except (asyncio.TimeoutError, Exception) as e:
                logger.error("Output stop error: %s", e)
        logger.info("Pipeline stopped.")


def print_commands(compose_mode: bool = False) -> None:
    """Print all available voice commands."""
    w1 = 24
    w2 = 38
    hr = "\u2500" * w1 + "\u253c" + "\u2500" * w2
    print()
    print("\u250c" + "\u2500" * w1 + "\u252c" + "\u2500" * w2 + "\u2510")
    print(f"\u2502 {'Hotkeys':<{w1-1}}\u2502 {'Action':<{w2-1}}\u2502")
    print("\u251c" + hr + "\u2524")
    print(f"\u2502 {'Shift+L':<{w1-1}}\u2502 {'Pause / resume listening':<{w2-1}}\u2502")
    print(f"\u2502 {'Ctrl+C':<{w1-1}}\u2502 {'Stop and exit':<{w2-1}}\u2502")
    print("\u251c" + hr + "\u2524")
    print(f"\u2502 {'Voice Commands':<{w1-1}}\u2502 {'Action':<{w2-1}}\u2502")
    print("\u251c" + hr + "\u2524")
    print(f"\u2502 {'stop listening':<{w1-1}}\u2502 {'Pause listening':<{w2-1}}\u2502")
    print(f"\u2502 {'switch to continuous':<{w1-1}}\u2502 {'Switch to continuous mode':<{w2-1}}\u2502")
    print(f"\u2502 {'switch to wake word':<{w1-1}}\u2502 {'Switch to wake word mode':<{w2-1}}\u2502")
    print(f"\u2502 {'...cancel':<{w1-1}}\u2502 {'Discard current utterance':<{w2-1}}\u2502")
    print(f"\u2502 {'help':<{w1-1}}\u2502 {'Show this help':<{w2-1}}\u2502")
    if compose_mode:
        print("\u251c" + hr + "\u2524")
        print(f"\u2502 {'Compose Commands':<{w1-1}}\u2502 {'Action':<{w2-1}}\u2502")
        print("\u251c" + hr + "\u2524")
        print(f"\u2502 {'send / end':<{w1-1}}\u2502 {'Flush buffer to handlers':<{w2-1}}\u2502")
        print(f"\u2502 {'undo / back':<{w1-1}}\u2502 {'Remove last utterance':<{w2-1}}\u2502")
        print(f"\u2502 {'redo':<{w1-1}}\u2502 {'Restore removed utterance':<{w2-1}}\u2502")
        print(f"\u2502 {'clear / cancel':<{w1-1}}\u2502 {'Clear entire buffer':<{w2-1}}\u2502")
        print(f"\u2502 {'resend':<{w1-1}}\u2502 {'Resend last output':<{w2-1}}\u2502")
        print(f"\u2502 {'preview':<{w1-1}}\u2502 {'Preview buffer contents':<{w2-1}}\u2502")
    print("\u2514" + "\u2500" * w1 + "\u2534" + "\u2500" * w2 + "\u2518")
    print()
