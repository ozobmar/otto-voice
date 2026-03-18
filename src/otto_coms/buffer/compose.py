"""Compose buffer — accumulates utterances for multi-sentence prompts."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from otto_coms.llm import LLMClient
    from otto_coms.handlers import OutputHandler

logger = logging.getLogger(__name__)


def _print_buffer(message: str) -> None:
    """Clear any in-place countdown line then print a buffer status message."""
    print("\r" + " " * 40 + "\r", end="", flush=True)
    print(message)


class ComposeBuffer:
    """Buffers utterances with undo/redo pointer and flushes as a single prompt."""

    def __init__(self, auto_send_delay: float = 8.0) -> None:
        self._utterances: list[str] = []
        self._pointer: int = 0
        self._auto_send_delay = auto_send_delay
        self._send_task: asyncio.Task | None = None
        self._llm: LLMClient | None = None
        self._outputs: list[OutputHandler] = []
        self._llm_cleanup_enabled: bool = True
        self._on_sent: Callable[[], None] | None = None
        self._last_sent: str | None = None

    def configure(
        self,
        llm: LLMClient | None,
        outputs: list[OutputHandler],
        llm_cleanup: bool = True,
    ) -> None:
        self._llm = llm
        self._outputs = outputs
        self._llm_cleanup_enabled = llm_cleanup

    def add(self, text: str) -> None:
        self._utterances = self._utterances[:self._pointer]
        self._utterances.append(text)
        self._pointer = len(self._utterances)
        self._reset_auto_send()
        self.display()

    def undo(self) -> None:
        if self._pointer > 0:
            self._pointer -= 1
            logger.info("Undo: pointer at %d/%d", self._pointer, len(self._utterances))
            self._reset_auto_send()
        else:
            logger.info("Undo: nothing to undo")
        self.display()

    def redo(self) -> None:
        if self._pointer < len(self._utterances):
            self._pointer += 1
            logger.info("Redo: pointer at %d/%d", self._pointer, len(self._utterances))
            self._reset_auto_send()
        else:
            logger.info("Redo: nothing to redo")
        self.display()

    def clear(self) -> None:
        self._cancel_auto_send()
        self._utterances.clear()
        self._pointer = 0
        _print_buffer("[BUFFER] Cleared")
        logger.info("Buffer cleared")

    def get_text(self) -> str:
        stripped = [u.rstrip(".!?,… ") for u in self._utterances[:self._pointer]]
        return " ".join(stripped)

    def is_empty(self) -> bool:
        return self._pointer == 0

    def display(self) -> None:
        if self.is_empty():
            _print_buffer("[BUFFER] (empty)")
            return
        combined = self.get_text()
        redo_count = len(self._utterances) - self._pointer
        suffix = f" [+{redo_count} undone]" if redo_count > 0 else ""
        _print_buffer(f"[BUFFER] {combined}{suffix}")

    async def preview(self) -> None:
        if self.is_empty():
            print("[PREVIEW] Nothing to preview")
            return

        text = self.get_text()

        if self._llm is None:
            print(f"[PREVIEW] (no LLM configured) {text}")
            return

        try:
            cleaned = await self._llm.cleanup(text)
            if cleaned and cleaned.strip():
                logger.info("Preview: '%s' -> '%s'", text, cleaned)
                self._utterances = [cleaned]
                self._pointer = 1
                print(f"[PREVIEW] {cleaned}")
            else:
                print(f"[PREVIEW] (LLM returned empty, keeping raw) {text}")
        except Exception as e:
            logger.warning("Preview LLM failed: %s", e)
            print(f"[PREVIEW] (LLM error, keeping raw) {text}")

    async def flush(self) -> None:
        self._cancel_auto_send()

        if self.is_empty():
            _print_buffer("[BUFFER] Nothing to send")
            return

        text = self.get_text()
        logger.info("Flushing buffer: '%s'", text)

        if self._llm_cleanup_enabled and self._llm is not None:
            try:
                cleaned = await self._llm.cleanup(text)
                if cleaned and cleaned.strip():
                    logger.info("LLM cleaned: '%s' -> '%s'", text, cleaned)
                    print(f"[LLM] {cleaned}")
                    text = cleaned
            except Exception as e:
                logger.warning("LLM cleanup failed, using raw text: %s", e)

        metadata = {"compose": True}
        for handler in self._outputs:
            try:
                await handler.emit(text, metadata)
            except Exception as e:
                logger.error("Output error (%s): %s", type(handler).__name__, e)

        self._last_sent = text
        self._utterances.clear()
        self._pointer = 0
        _print_buffer("[BUFFER] Sent and cleared")
        if self._on_sent is not None:
            self._on_sent()

    async def resend(self) -> None:
        if self._last_sent is None:
            _print_buffer("[BUFFER] Nothing to resend")
            return

        text = self._last_sent
        logger.info("Resending: '%s'", text)
        metadata = {"compose": True, "resend": True}

        for handler in self._outputs:
            try:
                await handler.emit(text, metadata)
            except Exception as e:
                logger.error("Output error (%s): %s", type(handler).__name__, e)

        _print_buffer("[BUFFER] Resent")
        if self._on_sent is not None:
            self._on_sent()

    def _reset_auto_send(self) -> None:
        self._cancel_auto_send()
        if self._auto_send_delay > 0:
            self._send_task = asyncio.create_task(self._auto_send())

    def _cancel_auto_send(self) -> None:
        if self._send_task is not None and not self._send_task.done():
            self._send_task.cancel()
            self._send_task = None

    async def _auto_send(self) -> None:
        try:
            remaining = int(self._auto_send_delay)
            while remaining > 0:
                print(f"\r[AUTO-SEND] Sending in {remaining}  ", end="", flush=True)
                await asyncio.sleep(1)
                remaining -= 1
            print(f"\r[AUTO-SEND] Sending           ", end="", flush=True)
        except asyncio.CancelledError:
            print("\r" + " " * 40 + "\r", end="", flush=True)
            return

        logger.info("Auto-send: buffer flush after %.1fs silence", self._auto_send_delay)

        # Clear own task reference before calling flush(),
        # otherwise flush() -> _cancel_auto_send() cancels us mid-flight.
        # Only clear if it still points to us — a new utterance arriving
        # between sleep completion and here would have replaced _send_task
        # with a fresh timer, and we must not overwrite that reference.
        current = asyncio.current_task()
        if self._send_task is current:
            self._send_task = None
        try:
            await self.flush()
        except Exception as e:
            logger.error("Auto-send failed: %s", e, exc_info=True)
