from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class WhisperConfig:
    model_size: str = "small"
    device: str = "auto"  # "auto" | "cpu" | "cuda"
    compute_type: str = "auto"  # forwarded to faster-whisper (e.g. "int8", "float16")
    language: str | None = None  # None = auto-detect


@dataclass
class STBConfig:
    encoder_checkpoint: Path = REPO_ROOT / "artifacts/pretrained_mix_6layer_BiLSTM/best/encoder.pt"
    transformer_checkpoint: Path = REPO_ROOT / "artifacts/pretrained_mix_6layer_BiLSTM/best/transformer.pt"
    detector_checkpoint: Path = REPO_ROOT / "artifacts/pretrained_mix_6layer_BiLSTM/best/detector.pt"
    device: str = "auto"
    threshold: float = 0.5
    local_max_window: int = 1  # +-k window used by local-max boundary decoding


@dataclass
class OllamaConfig:
    model: str = "qwen2.5:7b"
    host: str = "127.0.0.1"
    port: int = 11434
    auto_start: bool = True
    startup_timeout_s: float = 30.0
    request_timeout_s: float = 120.0

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"


@dataclass
class AppConfig:
    whisper: WhisperConfig = field(default_factory=WhisperConfig)
    stb: STBConfig = field(default_factory=STBConfig)
    ollama: OllamaConfig = field(default_factory=OllamaConfig)
