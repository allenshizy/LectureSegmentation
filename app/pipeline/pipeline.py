from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from app.config import AppConfig
from app.pipeline.asr import Segment, WhisperTranscriber
from app.pipeline.llm import OllamaClient
from app.pipeline.segmentation import StbSegmenter

logger = logging.getLogger(__name__)


@dataclass
class Chapter:
    start: float
    end: float
    text: str
    title: str | None = None


def _group_into_chapters(segments: list[Segment], boundary_times: list[float]) -> list[Chapter]:
    if not segments:
        return []

    boundaries = sorted(set(boundary_times))
    chapters: list[Chapter] = []
    cur_texts: list[str] = []
    cur_start = segments[0].start

    boundary_iter = iter(boundaries)
    next_boundary = next(boundary_iter, None)

    for seg in segments:
        cur_texts.append(seg.text)
        if next_boundary is not None and seg.end >= next_boundary:
            chapters.append(Chapter(start=cur_start, end=seg.end, text=" ".join(cur_texts)))
            cur_texts = []
            cur_start = seg.end
            next_boundary = next(boundary_iter, None)

    if cur_texts:
        chapters.append(Chapter(start=cur_start, end=segments[-1].end, text=" ".join(cur_texts)))

    return chapters


class SegmentationPipeline:
    """Whisper -> STB boundary detection -> optional LLM chapter titling."""

    def __init__(self, config: AppConfig | None = None) -> None:
        self.config = config or AppConfig()
        self.transcriber = WhisperTranscriber(self.config.whisper)
        self.segmenter = StbSegmenter(self.config.stb)
        self.llm = OllamaClient(self.config.ollama)

    def run(self, audio_path: str | Path, title_chapters: bool = True) -> list[Chapter]:
        segments = self.transcriber.transcribe(audio_path)
        boundary_times = self.segmenter.predict_boundaries(segments)
        chapters = _group_into_chapters(segments, boundary_times)

        if title_chapters:
            for chapter in chapters:
                try:
                    chapter.title = self.llm.title_chapter(chapter.text)
                except Exception:
                    logger.exception("Failed to title chapter [%.1f, %.1f]", chapter.start, chapter.end)
                    chapter.title = None

        return chapters
