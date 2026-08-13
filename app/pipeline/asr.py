from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from app.config import WhisperConfig

logger = logging.getLogger(__name__)


@dataclass
class Segment:
    """One ASR-decoded segment, treated as one "sentence" for STB."""

    start: float
    end: float
    text: str


class WhisperTranscriber:
    """Thin wrapper around faster-whisper, lazily loading the model."""

    def __init__(self, config: WhisperConfig | None = None) -> None:
        self.config = config or WhisperConfig()
        self._model = None

    def _load_model(self):
        if self._model is not None:
            return self._model
        from faster_whisper import WhisperModel

        device = self.config.device
        compute_type = self.config.compute_type
        if device == "auto":
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
        if compute_type == "auto":
            compute_type = "float16" if device == "cuda" else "int8"

        logger.info(
            "Loading faster-whisper model=%s device=%s compute_type=%s",
            self.config.model_size,
            device,
            compute_type,
        )
        self._model = WhisperModel(self.config.model_size, device=device, compute_type=compute_type)
        return self._model

    def transcribe(self, audio_path: str | Path) -> list[Segment]:
        """Transcribe an audio/video file into a flat list of timed segments."""

        model = self._load_model()
        segments, _info = model.transcribe(str(audio_path), language=self.config.language)
        result = [
            Segment(start=float(seg.start), end=float(seg.end), text=seg.text.strip())
            for seg in segments
            if seg.text and seg.text.strip()
        ]
        logger.info("Transcribed %s into %d segments", audio_path, len(result))
        return result
