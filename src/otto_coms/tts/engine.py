"""Local TTS engine — converts text to speech audio.

Uses edge-tts (Microsoft Edge online TTS) with a dedicated worker thread
and queue for non-blocking speech. MP3 output is decoded via miniaudio
and played through sounddevice.

Supports barge-in: external code can call interrupt() to stop playback
immediately and clear the speech queue.

Sentence-level streaming: long responses are split into sentences and
each sentence is synthesised and played independently. The first sentence
starts playing while remaining sentences are queued for synthesis.
"""

from __future__ import annotations

import asyncio
import logging
import queue
import re
import threading

import numpy as np
import sounddevice as sd

logger = logging.getLogger(__name__)

# Sentinel to signal the worker thread to stop
_STOP = object()


class TTSEngine:
    """Text-to-speech using edge-tts.

    All speech is serialised through a single worker thread that runs
    its own asyncio event loop for the async edge-tts API.

    Barge-in support: call interrupt() to stop current playback and
    clear queued speech. Check is_playing to know if audio is active.
    """

    def __init__(
        self,
        voice: str = "en-IE-EmilyNeural",
        rate: str = "+0%",
        volume: str = "+0%",
    ) -> None:
        self._voice = voice
        self._rate = rate
        self._volume = volume
        self._available = False
        self._queue: queue.Queue = queue.Queue()
        self._thread: threading.Thread | None = None
        self._playing = threading.Event()  # set while audio is playing
        self._interrupted = threading.Event()  # set to signal barge-in
        self._pending: int = 0  # items queued but not yet finished

    def load(self) -> bool:
        """Verify edge-tts is importable and start the worker thread."""
        try:
            import edge_tts  # noqa: F401
            import miniaudio  # noqa: F401

            self._available = True
            self._thread = threading.Thread(target=self._worker, daemon=True)
            self._thread.start()
            logger.info("TTS engine loaded (voice=%s, rate=%s)", self._voice, self._rate)
            return True
        except ImportError as e:
            logger.error("TTS engine load failed — missing package: %s", e)
            return False
        except Exception as e:
            logger.error("TTS engine load failed: %s", e)
            return False

    @property
    def available(self) -> bool:
        return self._available

    @property
    def is_playing(self) -> bool:
        """True while TTS audio is actively playing through speakers."""
        return self._playing.is_set()

    def speak(self, text: str) -> None:
        """Queue text for speech. Non-blocking — returns immediately."""
        if not self._available:
            return
        self._pending += 1
        self._queue.put(text)

    def interrupt(self) -> None:
        """Barge-in: stop current playback and clear queued speech."""
        if not self._available:
            return
        self._interrupted.set()
        # Drain the queue so nothing else plays after the interrupt
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
        # Stop sounddevice playback immediately
        try:
            sd.stop()
        except Exception:
            pass
        logger.info("TTS interrupted (barge-in)")

    def _worker(self) -> None:
        """Process speech queue on a dedicated thread.

        Runs its own asyncio event loop for edge-tts async calls.
        """
        loop = asyncio.new_event_loop()

        try:
            while True:
                item = self._queue.get()
                if item is _STOP:
                    break
                # Check if interrupted while waiting in queue
                if self._interrupted.is_set():
                    self._interrupted.clear()
                    continue
                try:
                    logger.debug("TTS speaking: '%s'", item[:80])
                    loop.run_until_complete(self._synthesise_and_play(item))
                    logger.debug("TTS done: '%s'", item[:80])
                except Exception as e:
                    logger.error("TTS speak error: %s", e, exc_info=True)
                finally:
                    self._pending = max(0, self._pending - 1)
                    self._playing.clear()
                    self._interrupted.clear()
        finally:
            loop.close()

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        """Split text into sentences for incremental TTS.

        Splits on sentence-ending punctuation followed by whitespace.
        Short texts (single sentence) are returned as-is.
        """
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        return [s for s in sentences if s.strip()]

    async def _synthesise_and_play(self, text: str) -> None:
        """Generate speech audio with edge-tts and play it.

        For multi-sentence text, splits into sentences and plays each
        independently — the first sentence starts playing while remaining
        sentences are still being synthesised. Single sentences play
        directly.

        Checks _interrupted flag throughout to support barge-in.
        """
        if self._interrupted.is_set():
            return

        sentences = self._split_sentences(text)
        if not sentences:
            return

        self._playing.set()
        try:
            for sentence in sentences:
                if self._interrupted.is_set():
                    return
                await self._synthesise_and_play_chunk(sentence)
        finally:
            self._playing.clear()

    async def _synthesise_and_play_chunk(self, text: str) -> None:
        """Synthesise and play a single chunk of text."""
        import edge_tts
        import miniaudio

        if self._interrupted.is_set():
            return

        communicate = edge_tts.Communicate(
            text, voice=self._voice, rate=self._rate, volume=self._volume,
        )
        mp3_chunks: list[bytes] = []
        async for chunk in communicate.stream():
            if self._interrupted.is_set():
                logger.debug("TTS synthesis interrupted mid-stream")
                return
            if chunk["type"] == "audio":
                mp3_chunks.append(chunk["data"])

        if not mp3_chunks:
            logger.warning("TTS produced no audio for: '%s'", text[:80])
            return

        if self._interrupted.is_set():
            return

        mp3_data = b"".join(mp3_chunks)

        # Decode MP3 → PCM
        decoded = miniaudio.decode(mp3_data, output_format=miniaudio.SampleFormat.SIGNED16)
        audio = np.frombuffer(decoded.samples, dtype=np.int16)

        # Convert to mono if multi-channel
        if decoded.nchannels > 1:
            audio = audio.reshape(-1, decoded.nchannels).mean(axis=1).astype(np.int16)

        # Normalise to float32 [-1, 1] for sounddevice
        audio_float = audio.astype(np.float32) / 32768.0

        # Play through speakers — non-blocking so we can check for interrupts
        try:
            samplerate = decoded.sample_rate
            out_device = sd.default.device[1] if sd.default.device[1] is not None else None

            # Resample to the output device's native rate if different
            if out_device is not None:
                device_info = sd.query_devices(out_device, kind="output")
                native_rate = int(device_info["default_samplerate"])
                if native_rate != samplerate:
                    ratio = native_rate / samplerate
                    n_out = int(len(audio_float) * ratio)
                    indices = np.arange(n_out) / ratio
                    audio_float = np.interp(indices, np.arange(len(audio_float)), audio_float).astype(np.float32)
                    samplerate = native_rate

            sd.play(audio_float, samplerate=samplerate, device=out_device, blocking=False)

            # Wait for playback to finish, checking for barge-in every 50ms
            duration_s = len(audio_float) / samplerate
            elapsed = 0.0
            poll_interval = 0.05
            while elapsed < duration_s + 0.1:
                if self._interrupted.is_set():
                    sd.stop()
                    logger.debug("TTS playback interrupted by barge-in")
                    return
                await asyncio.sleep(poll_interval)
                elapsed += poll_interval
        except Exception as e:
            logger.error("TTS playback error: %s", e)

    def wait(self, timeout: float = 60.0) -> None:
        """Block until TTS has finished playing (for use in executor threads)."""
        import time
        start = time.monotonic()
        while self._pending > 0 or self._playing.is_set():
            if time.monotonic() - start > timeout:
                break
            time.sleep(0.05)

    def stop(self) -> None:
        """Stop the worker thread."""
        if self._thread is not None and self._thread.is_alive():
            self._queue.put(_STOP)
            self._thread.join(timeout=5)
