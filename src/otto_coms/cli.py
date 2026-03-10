"""CLI entry point and argument parsing."""

from __future__ import annotations

import argparse
import asyncio
import logging
import logging.handlers
import sys
from pathlib import Path

import sounddevice as sd

from otto_coms.config import Config, apply_cli_overrides, load_config
from otto_coms.processing.hardware import detect_hardware, recommend_settings

logger = logging.getLogger(__name__)


def _list_devices() -> None:
    """Print available audio input devices and exit."""
    print("\nAvailable audio input devices:\n")
    devices = sd.query_devices()
    for i, dev in enumerate(devices):
        if dev["max_input_channels"] > 0:
            default = " (DEFAULT)" if i == sd.default.device[0] else ""
            print(f"  [{i}] {dev['name']}{default}")
            print(f"       Channels: {dev['max_input_channels']}, "
                  f"Sample Rate: {dev['default_samplerate']:.0f} Hz")
    print()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="otto-coms",
        description="Unified voice client for Otto — STT, TTS, wake word, compose mode",
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--list-devices", action="store_true",
        help="List available audio input devices and exit",
    )
    parser.add_argument(
        "--device", type=int, default=None,
        help="Audio input device index (see --list-devices)",
    )
    parser.add_argument(
        "--gain", type=float, default=None,
        help="Mic gain multiplier (e.g. 1.5 = +50%%, 2.0 = double)",
    )
    parser.add_argument(
        "--model", type=str, default=None,
        choices=["auto", "tiny", "base", "small", "medium", "large-v3"],
        help="Whisper model size",
    )
    parser.add_argument(
        "--stt-device", type=str, default=None,
        choices=["auto", "cpu", "cuda"],
        help="Device for STT inference",
    )
    parser.add_argument(
        "--language", type=str, default=None,
        help="Language code (e.g. 'en'). Omit for auto-detect",
    )
    parser.add_argument(
        "--outputs", nargs="+", default=None,
        choices=["console", "file", "clipboard", "websocket", "otto-api", "cc-direct"],
        help="Output handlers to enable",
    )
    parser.add_argument(
        "--compose", action="store_true",
        help="Enable compose mode (buffer utterances, say 'send' to flush)",
    )
    parser.add_argument(
        "--no-auto-send", action="store_true",
        help="Disable auto-send in compose mode",
    )
    parser.add_argument(
        "--listen", type=str, default=None,
        choices=["continuous", "wake-word"],
        help="Listening mode",
    )
    parser.add_argument(
        "--transmission", type=str, default=None,
        choices=["sync", "async"],
        help="Transmission mode for Otto API",
    )
    parser.add_argument(
        "--ww-model", type=str, default=None,
        help="Wake word model name or path to ONNX file",
    )
    parser.add_argument(
        "--ww-threshold", type=float, default=None,
        help="Wake word detection threshold (0.0-1.0)",
    )
    parser.add_argument(
        "--otto-url", type=str, default=None,
        help="Otto server URL (e.g. http://otto-core-01:8080)",
    )
    parser.add_argument(
        "--cc-session", type=str, default=None,
        help="Resume an existing Claude Code session by ID (cc-direct only)",
    )
    parser.add_argument(
        "--cc-dir", type=str, default=None,
        help="Working directory for Claude Code (cc-direct only, defaults to cwd)",
    )
    parser.add_argument(
        "--tts", action="store_true", default=None,
        help="Enable TTS (speak responses)",
    )
    parser.add_argument(
        "--no-tts", action="store_true", default=None,
        help="Disable TTS",
    )
    parser.add_argument(
        "--tts-rate", type=str, default=None,
        help="TTS speech rate (e.g. '+30%%', '-10%%', '+0%%')",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable debug logging",
    )
    parser.add_argument(
        "--quiet", "-q", action="store_true",
        help="Suppress info messages, show warnings and errors only",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    # Logging — console at WARNING (-q), DEBUG (-v), or INFO (default)
    if args.quiet:
        console_level = logging.WARNING
    elif args.verbose:
        console_level = logging.DEBUG
    else:
        console_level = logging.INFO
    log_fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)
    console_handler.setFormatter(log_fmt)
    root.addHandler(console_handler)

    # Rotating file handler
    log_dir = Path(__file__).resolve().parent.parent.parent / "logs"
    log_dir.mkdir(exist_ok=True)
    file_handler = logging.handlers.RotatingFileHandler(
        log_dir / "otto-coms.log",
        maxBytes=5 * 1024 * 1024,
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    ))
    root.addHandler(file_handler)
    logger.info("Log file: %s", log_dir / "otto-coms.log")

    if args.list_devices:
        _list_devices()
        sys.exit(0)

    # Load config
    config_path = Path(args.config) if args.config else None
    config: Config = load_config(config_path)
    config = apply_cli_overrides(config, args)

    # Hardware detection and auto-config
    hw = detect_hardware()
    logger.info("Hardware: %s", hw)

    if config.stt.model == "auto" or config.stt.device == "auto" or config.stt.compute_type == "auto":
        rec = recommend_settings(hw)
        if config.stt.model == "auto":
            config.stt.model = rec["model"]
        if config.stt.device == "auto":
            config.stt.device = rec["device"]
        if config.stt.compute_type == "auto":
            config.stt.compute_type = rec["compute_type"]
        logger.info(
            "STT config: model=%s, device=%s, compute_type=%s",
            config.stt.model, config.stt.device, config.stt.compute_type,
        )

    # Run pipeline
    from otto_coms.pipeline import run_pipeline

    try:
        asyncio.run(run_pipeline(config))
    except KeyboardInterrupt:
        pass  # Pipeline prints "Stopping..." in its finally block
