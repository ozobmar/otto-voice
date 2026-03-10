"""cc-direct output handler — pipes transcription to a local Claude Code instance."""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import TYPE_CHECKING

from otto_coms.handlers import OutputHandler

if TYPE_CHECKING:
    from otto_coms.config import Config

logger = logging.getLogger(__name__)


class CcDirectOutput(OutputHandler):
    """Send transcribed text to a local Claude Code instance via claude -p."""

    def __init__(self, config: Config) -> None:
        self._working_dir = config.output_settings.cc_direct.working_dir
        self._model = config.output_settings.cc_direct.model
        self._system_prompt = config.output_settings.cc_direct.system_prompt
        self._max_budget = config.output_settings.cc_direct.max_budget_usd
        self._session_id: str | None = config.output_settings.cc_direct.session_id
        self._on_response: callable | None = None

    def set_response_callback(self, callback: callable) -> None:
        """Set a callback for handling responses (used by pipeline for TTS)."""
        self._on_response = callback

    async def start(self) -> None:
        logger.info("cc-direct output starting (working_dir=%s)", self._working_dir)

    async def emit(self, text: str, metadata: dict | None = None) -> None:
        if not text.strip():
            return

        if self._session_id is None:
            response = await self._call_claude(text, json_output=True)
            if response is None:
                return
            try:
                data = json.loads(response)
                self._session_id = data.get("session_id")
                result = data.get("result", "")
                logger.info("cc-direct session started: %s", self._session_id)
                print(f"[CC] {result}")
                if result and self._on_response is not None:
                    self._on_response({"response": result})
            except (json.JSONDecodeError, KeyError) as e:
                logger.error("Failed to parse session init response: %s", e)
        else:
            response = await self._call_claude(text, json_output=False)
            if response is not None:
                print(f"[CC] {response}")
                if self._on_response is not None:
                    self._on_response({"response": response})

    async def _call_claude(self, text: str, json_output: bool) -> str | None:
        cmd = ["claude", "-p", "--permission-mode", "acceptEdits"]

        if json_output:
            cmd.extend(["--output-format", "json"])
        else:
            cmd.extend(["--resume", self._session_id])

        if self._model:
            cmd.extend(["--model", self._model])

        if self._system_prompt:
            cmd.extend(["--system-prompt", self._system_prompt])

        if self._max_budget:
            cmd.extend(["--max-budget-usd", str(self._max_budget)])

        cmd.append(text)

        proc = None
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self._working_dir,
                env={**os.environ, "CLAUDECODE": ""},  # inherit env, unset CLAUDECODE to avoid nested session error
            )
            stdout, stderr = await proc.communicate()

            if proc.returncode != 0:
                logger.error("claude -p failed (rc=%d): %s", proc.returncode, stderr.decode().strip())
                return None

            return stdout.decode().strip()

        except asyncio.CancelledError:
            if proc is not None and proc.returncode is None:
                proc.kill()
                await proc.wait()
            raise
        except FileNotFoundError:
            logger.error("claude CLI not found — is it installed and on PATH?")
            return None
        except Exception as e:
            logger.error("cc-direct error: %s", e)
            return None

    async def stop(self) -> None:
        if self._session_id:
            logger.info("cc-direct session ended: %s", self._session_id)
