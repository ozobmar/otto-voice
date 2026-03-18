"""Voice command handling — detects and executes voice commands from transcriptions."""

from __future__ import annotations

import asyncio
import logging
import re
from typing import TYPE_CHECKING

from otto_coms.platform.input_sim import press_enter

if TYPE_CHECKING:
    from otto_coms.buffer.compose import ComposeBuffer

logger = logging.getLogger(__name__)


# Non-compose mode commands: phrase -> (action, description)
_COMMANDS: dict[str, tuple[callable, str]] = {
    "send": (press_enter, "Enter key"),
    "transmit": (press_enter, "Enter key"),
    "execute": (press_enter, "Enter key"),
    "enter": (press_enter, "Enter key"),
    "run": (press_enter, "Enter key"),
    "end": (press_enter, "Enter key"),
}

# Compose mode command sets
_COMPOSE_SEND = {"send", "end"}
_COMPOSE_CLEAR = {"clear", "cancel"}
_COMPOSE_UNDO = {"undo", "back"}
_COMPOSE_REDO = {"redo"}
_COMPOSE_RESEND = {"resend"}
_COMPOSE_PREVIEW = {"preview"}

# Mode switching commands
_MODE_COMMANDS = {
    "switch to continuous": "listen_continuous",
    "continuous mode": "listen_continuous",
    "switch to wake word": "listen_wake_word",
    "wake word mode": "listen_wake_word",
    "stop listening": "pause",
    # "start listening" removed — can't detect voice commands when paused.
    # Resume via wake word (ONNX stays active) or Shift+L hotkey.
    "sync mode": "tx_sync",
    "switch to sync": "tx_sync",
    "async mode": "tx_async",
    "switch to async": "tx_async",
    "help": "show_commands",
}


class CommandResult:
    """Result of a voice command check."""

    def __init__(
        self,
        handled: bool = False,
        text: str | None = None,
        mode_switch: str | None = None,
    ) -> None:
        self.handled = handled
        self.text = text  # None = suppress output
        self.mode_switch = mode_switch  # signal to pipeline


def check_voice_command(
    text: str,
    buffer: ComposeBuffer | None = None,
) -> CommandResult:
    """Check if transcribed text is a voice command.

    Returns CommandResult indicating whether the command was handled.
    """
    normalised = re.sub(r'[.?!]+$', '', text.strip().lower()).strip()

    # Mode switching (always checked first)
    mode_action = _MODE_COMMANDS.get(normalised)
    if mode_action is not None:
        logger.info("Mode command: '%s' -> %s", normalised, mode_action)
        return CommandResult(handled=True, mode_switch=mode_action)

    # Compose mode commands
    if buffer is not None:
        return _check_compose_command(normalised, buffer, text)

    # Non-compose commands
    entry = _COMMANDS.get(normalised)
    if entry is not None:
        action, description = entry
        logger.info("Voice command: '%s' -> %s", normalised, description)
        action()
        return CommandResult(handled=True)

    # Trailing "cancel": discard utterance
    if normalised.endswith("cancel"):
        logger.info("Voice command: cancel -> discarding utterance")
        return CommandResult(handled=True)

    return CommandResult(handled=False, text=text)


def _check_compose_command(
    normalised: str,
    buffer: ComposeBuffer,
    text: str,
) -> CommandResult:
    """Handle commands in compose mode."""
    if normalised in _COMPOSE_SEND:
        logger.info("Compose command: '%s' -> flush buffer", normalised)
        asyncio.create_task(buffer.flush())
        return CommandResult(handled=True)

    if normalised in _COMPOSE_CLEAR:
        logger.info("Compose command: '%s' -> clear buffer", normalised)
        buffer.clear()
        return CommandResult(handled=True)

    if normalised in _COMPOSE_UNDO:
        logger.info("Compose command: '%s' -> undo", normalised)
        buffer.undo()
        return CommandResult(handled=True)

    if normalised in _COMPOSE_REDO:
        logger.info("Compose command: '%s' -> redo", normalised)
        buffer.redo()
        return CommandResult(handled=True)

    if normalised in _COMPOSE_RESEND:
        logger.info("Compose command: '%s' -> resend", normalised)
        asyncio.create_task(buffer.resend())
        return CommandResult(handled=True)

    if normalised in _COMPOSE_PREVIEW:
        logger.info("Compose command: '%s' -> preview", normalised)
        asyncio.create_task(buffer.preview())
        return CommandResult(handled=True)

    if normalised.endswith("cancel"):
        logger.info("Compose command: cancel -> discarding utterance")
        return CommandResult(handled=True)

    # Trailing send: "some text send" -> add remainder to buffer then flush
    for cmd in _COMPOSE_SEND:
        if normalised.endswith(" " + cmd):
            remainder = normalised[: -(len(cmd) + 1)].strip()
            if remainder:
                buffer.add(remainder)
            logger.info("Compose command: trailing '%s' -> flush buffer (remainder: '%s')", cmd, remainder)
            asyncio.create_task(buffer.flush())
            return CommandResult(handled=True)

    return CommandResult(handled=False, text=text)
