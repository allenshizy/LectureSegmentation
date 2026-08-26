from __future__ import annotations

import json
import logging
import platform
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from app.config import AppConfig
from app.pipeline.asr import Segment, WhisperTranscriber
from app.pipeline.llm import OllamaClient
from app.pipeline.segmentation import StbSegmenter

logger = logging.getLogger(__name__)

SEGMENTED_ARTIFACTS_DIR = Path(__file__).resolve().parents[2] / "artifacts" / "segmented"


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

    def _save_result(
        self,
        audio_path: str | Path,
        result: PipelineResult,
        *,
        describe_chapters: bool,
        duration_seconds: float,
    ) -> Path:
        source = Path(audio_path).resolve()
        timestamp = datetime.now(timezone.utc)
        artifact = {
            "summary": {
                "created_at": timestamp.isoformat(),
                "source_file": str(source),
                "duration_seconds": round(duration_seconds, 3),
                "generate_descriptions": describe_chapters,
                "chapter_count": len(result.chapters),
                "models": {
                    "asr": {
                        "name": "faster-whisper",
                        "model": self.config.whisper.model_size,
                        "device": self.config.whisper.device,
                        "compute_type": self.config.whisper.compute_type,
                        "language": self.config.whisper.language,
                    },
                    "segmentation": {
                        "name": "STB",
                        "transformer_checkpoint": str(self.config.stb.transformer_checkpoint),
                        "detector_checkpoint": str(self.config.stb.detector_checkpoint),
                        "device": self.config.stb.device,
                        "threshold": self.config.stb.threshold,
                        "local_max_window": self.config.stb.local_max_window,
                    },
                    "description": {
                        "name": "Ollama",
                        "model": self.config.ollama.model if describe_chapters else None,
                    },
                },
                "system": {
                    "platform": platform.platform(),
                    "python_version": platform.python_version(),
                },
            },
            "course_summary": result.course_summary,
            "chapters": [
                {
                    "start": chapter.start,
                    "end": chapter.end,
                    "title": chapter.title,
                    "keywords": chapter.keywords,
                    "summary": chapter.summary,
                    "transcript": chapter.text,
                    "description_error": chapter.description_error,
                }
                for chapter in result.chapters
            ],
        }
        SEGMENTED_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        output_path = SEGMENTED_ARTIFACTS_DIR / f"{source.stem}_{timestamp:%Y%m%dT%H%M%S%fZ}.json"
        output_path.write_text(json.dumps(artifact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        logger.info("Saved segmentation result to %s", output_path)
        return output_path

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

        result = PipelineResult(chapters=chapters, course_summary=course_summary)
        duration_seconds = time.perf_counter() - started_at
        artifact_path = self._save_result(
            audio_path,
            result,
            describe_chapters=describe_chapters,
            duration_seconds=duration_seconds,
        )
        logger.info("Segmentation pipeline finished in %.1fs; result saved to %s", duration_seconds, artifact_path)
        return result
