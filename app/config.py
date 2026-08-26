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
    # encoder is un-finetuned SBERT (all-MiniLM-L6-v2); rebuilt from HF, not checked into git.
    transformer_checkpoint: Path = REPO_ROOT / "app/checkpoints/transformer.pt"
    detector_checkpoint: Path = REPO_ROOT / "app/checkpoints/detector.pt"
    device: str = "auto"
    threshold: float = 0.75  # calibrated in artifacts/pretrained_mix_6layer_BiLSTM/summary.json
    local_max_window: int = 1  # +-k window used by local-max boundary decoding


@dataclass
class OllamaConfig:
    model: str = "qwen3:4b"
    host: str = "127.0.0.1"
    port: int = 11434
    auto_start: bool = True
    auto_pull: bool = True
    startup_timeout_s: float = 30.0
    pull_timeout_s: float = 1800.0  # first pull can be a large download
    tags_timeout_s: float = 15.0
    request_timeout_s: float = 120.0

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"


@dataclass
class AppConfig:
    whisper: WhisperConfig = field(default_factory=WhisperConfig)
    stb: STBConfig = field(default_factory=STBConfig)
    ollama: OllamaConfig = field(default_factory=OllamaConfig)
