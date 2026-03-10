"""Configuration loading and dataclass definitions."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

_DEFAULT_CONFIG = Path(__file__).resolve().parent.parent.parent / "config.default.yaml"


@dataclass
class AudioConfig:
    device: int | str | None = None
    sample_rate: int = 16000
    block_size: int = 512
    gain: float = 1.0


@dataclass
class VADConfig:
    threshold: float = 0.5
    silence_duration_ms: int = 700
    min_speech_duration_ms: int = 250
    speech_pad_ms: int = 250


@dataclass
class STTConfig:
    model: str = "auto"
    device: str = "auto"
    compute_type: str = "auto"
    language: str | None = None
    beam_size: int = 5
    cpu_threads: int = 0


@dataclass
class FileOutputConfig:
    path: str = "transcriptions.txt"
    mode: str = "append"


@dataclass
class ClipboardOutputConfig:
    paste: bool = True
    paste_delay_ms: int = 50
    auto_send: bool = True
    auto_send_delay_ms: int = 2000


@dataclass
class WebSocketOutputConfig:
    host: str = "localhost"
    port: int = 8765


@dataclass
class OttoApiOutputConfig:
    url: str = "http://otto-core-01:8080"
    timeout: int = 60
    return_audio: bool = True
    voice: str = "default"
    provider: str = "ollama"


@dataclass
class CcDirectOutputConfig:
    working_dir: str | None = None
    model: str | None = None
    system_prompt: str | None = None
    max_budget_usd: float | None = None
    session_id: str | None = None


@dataclass
class OutputSettings:
    file: FileOutputConfig = field(default_factory=FileOutputConfig)
    clipboard: ClipboardOutputConfig = field(default_factory=ClipboardOutputConfig)
    websocket: WebSocketOutputConfig = field(default_factory=WebSocketOutputConfig)
    otto_api: OttoApiOutputConfig = field(default_factory=OttoApiOutputConfig)
    cc_direct: CcDirectOutputConfig = field(default_factory=CcDirectOutputConfig)


@dataclass
class OllamaConfig:
    base_url: str = "http://localhost:11434"
    model: str = "llama3.2"


@dataclass
class ClaudeConfig:
    base_url: str = "http://localhost:3000"
    model: str = "claude-sonnet-4-20250514"


@dataclass
class LLMConfig:
    enabled: bool = False
    provider: str = "ollama"
    ollama: OllamaConfig = field(default_factory=OllamaConfig)
    claude: ClaudeConfig = field(default_factory=ClaudeConfig)


@dataclass
class ComposeConfig:
    enabled: bool = False
    auto_send_delay_ms: int = 8000
    llm_cleanup: bool = True


@dataclass
class WakeWordConfig:
    model: str = "hey_jarvis"
    threshold: float = 0.5
    timeout_seconds: int = 60
    feedback_beep: bool = True


@dataclass
class ListeningConfig:
    mode: str = "continuous"  # "continuous" or "wake_word"
    wake_word: WakeWordConfig = field(default_factory=WakeWordConfig)


@dataclass
class TTSConfig:
    enabled: bool = True
    voice: str = "en-IE-EmilyNeural"  # edge-tts voice name
    rate: str = "+30%"    # edge-tts rate adjustment (e.g. "+10%", "-20%")
    volume: str = "+0%"  # edge-tts volume adjustment


@dataclass
class TransmissionConfig:
    mode: str = "sync"  # "sync" or "async"
    async_callback_host: str = "0.0.0.0"
    async_callback_port: int = 8766


@dataclass
class Config:
    audio: AudioConfig = field(default_factory=AudioConfig)
    vad: VADConfig = field(default_factory=VADConfig)
    stt: STTConfig = field(default_factory=STTConfig)
    outputs: list[str] = field(default_factory=lambda: ["console"])
    output_settings: OutputSettings = field(default_factory=OutputSettings)
    compose: ComposeConfig = field(default_factory=ComposeConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    listening: ListeningConfig = field(default_factory=ListeningConfig)
    tts: TTSConfig = field(default_factory=TTSConfig)
    transmission: TransmissionConfig = field(default_factory=TransmissionConfig)


def _deep_merge(base: dict, override: dict) -> dict:
    """Merge override into base recursively."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _dict_to_config(data: dict[str, Any]) -> Config:
    """Convert a nested dict to a Config dataclass."""
    audio = AudioConfig(**data.get("audio", {}))
    vad = VADConfig(**data.get("vad", {}))
    stt = STTConfig(**data.get("stt", {}))
    outputs = data.get("outputs", ["console"])

    os_data = data.get("output_settings", {})
    output_settings = OutputSettings(
        file=FileOutputConfig(**os_data.get("file", {})),
        clipboard=ClipboardOutputConfig(**os_data.get("clipboard", {})),
        websocket=WebSocketOutputConfig(**os_data.get("websocket", {})),
        otto_api=OttoApiOutputConfig(**os_data.get("otto_api", {})),
    )

    compose = ComposeConfig(**data.get("compose", {}))

    llm_data = data.get("llm", {})
    llm = LLMConfig(
        enabled=llm_data.get("enabled", False),
        provider=llm_data.get("provider", "ollama"),
        ollama=OllamaConfig(**llm_data.get("ollama", {})),
        claude=ClaudeConfig(**llm_data.get("claude", {})),
    )

    listen_data = data.get("listening", {})
    ww_data = listen_data.get("wake_word", {})
    listening = ListeningConfig(
        mode=listen_data.get("mode", "continuous"),
        wake_word=WakeWordConfig(**ww_data),
    )

    tts = TTSConfig(**data.get("tts", {}))

    tx_data = data.get("transmission", {})
    transmission = TransmissionConfig(
        mode=tx_data.get("mode", "sync"),
        async_callback_host=tx_data.get("async_callback_host", "0.0.0.0"),
        async_callback_port=tx_data.get("async_callback_port", 8766),
    )

    return Config(
        audio=audio,
        vad=vad,
        stt=stt,
        outputs=outputs,
        output_settings=output_settings,
        compose=compose,
        llm=llm,
        listening=listening,
        tts=tts,
        transmission=transmission,
    )


def load_config(config_path: Path | None = None) -> Config:
    """Load config from YAML file, falling back to defaults."""
    base: dict[str, Any] = {}

    if _DEFAULT_CONFIG.exists():
        with open(_DEFAULT_CONFIG) as f:
            base = yaml.safe_load(f) or {}

    if config_path and config_path.exists():
        with open(config_path) as f:
            user = yaml.safe_load(f) or {}
        base = _deep_merge(base, user)

    return _dict_to_config(base)


def apply_cli_overrides(config: Config, args: Any) -> Config:
    """Apply CLI argument overrides to config."""
    if getattr(args, "device", None) is not None:
        config.audio.device = args.device
    if getattr(args, "gain", None) is not None:
        config.audio.gain = args.gain
    if getattr(args, "model", None) is not None:
        config.stt.model = args.model
    if getattr(args, "stt_device", None) is not None:
        config.stt.device = args.stt_device
    if getattr(args, "language", None) is not None:
        config.stt.language = args.language
    if getattr(args, "outputs", None) is not None:
        config.outputs = args.outputs
    if getattr(args, "compose", False):
        config.compose.enabled = True
    if getattr(args, "no_auto_send", False):
        config.compose.auto_send_delay_ms = 0
    if getattr(args, "listen", None) is not None:
        config.listening.mode = args.listen.replace("-", "_")
    if getattr(args, "transmission", None) is not None:
        config.transmission.mode = args.transmission
    if getattr(args, "ww_model", None) is not None:
        config.listening.wake_word.model = args.ww_model
    if getattr(args, "ww_threshold", None) is not None:
        config.listening.wake_word.threshold = args.ww_threshold
    if getattr(args, "otto_url", None) is not None:
        config.output_settings.otto_api.url = args.otto_url
    if getattr(args, "cc_session", None) is not None:
        config.output_settings.cc_direct.session_id = args.cc_session
    if getattr(args, "cc_dir", None) is not None:
        config.output_settings.cc_direct.working_dir = args.cc_dir
    if getattr(args, "tts", None):
        config.tts.enabled = True
    if getattr(args, "no_tts", None):
        config.tts.enabled = False
    if getattr(args, "tts_rate", None) is not None:
        config.tts.rate = args.tts_rate
    return config
