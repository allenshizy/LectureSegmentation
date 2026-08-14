from __future__ import annotations

import logging
import time
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
    keywords: list[str] | None = None
    summary: str | None = None
    description_error: str | None = None


@dataclass
class PipelineResult:
    chapters: list[Chapter]
    course_summary: str | None = None


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

    def run(self, audio_path: str | Path, describe_chapters: bool = True) -> PipelineResult:
        logger.info("Segmentation pipeline started for %s", audio_path)
        started_at = time.perf_counter()
        segments = self.transcriber.transcribe(audio_path)
        boundary_times = self.segmenter.predict_boundaries(segments)
        chapters = _group_into_chapters(segments, boundary_times)
        logger.info("Grouped %d segments into %d chapters", len(segments), len(chapters))

        course_summary = None
        if describe_chapters:
            logger.info("Qwen chapter descriptions started for %d chapters", len(chapters))
            for chapter_index, chapter in enumerate(chapters, start=1):
                try:
                    logger.info("Describing chapter %d/%d", chapter_index, len(chapters))
                    described = self.llm.describe_chapter(chapter.text)
                    chapter.title = described["title"]
                    chapter.keywords = described["keywords"]
                    chapter.summary = described["summary"]
                except Exception as exc:
                    chapter.description_error = str(exc)
                    logger.exception("Failed to describe chapter [%.1f, %.1f]", chapter.start, chapter.end)

            successful_summaries = [chapter.summary for chapter in chapters if chapter.summary]
            if successful_summaries:
                try:
                    logger.info("Generating course overview from %d chapter summaries", len(successful_summaries))
                    course_summary = self.llm.summarize_course(successful_summaries)
                    logger.info("Course overview generated")
                except Exception:
                    logger.exception("Failed to generate course overview")
            else:
                logger.warning("Skipping course overview because no chapter summaries were generated")

            logger.info("Segmentation pipeline finished in %.1fs", time.perf_counter() - started_at)
        return PipelineResult(chapters=chapters, course_summary=course_summary)
