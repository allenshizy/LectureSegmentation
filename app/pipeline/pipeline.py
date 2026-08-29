from __future__ import annotations

import json
import logging
import platform
import re
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from app.config import AppConfig
from app.pipeline.asr import Segment, WhisperTranscriber
from app.pipeline.llm import OllamaClient
from app.pipeline.segmentation import StbSegmenter
from app.pipeline.youtube import YouTubeInfo, download_youtube_audio, get_youtube_info

logger = logging.getLogger(__name__)

SEGMENTED_ARTIFACTS_DIR = Path(__file__).resolve().parents[2] / "artifacts" / "segmented"
YOUTUBE_AUDIO_CACHE_DIR = Path(__file__).resolve().parents[2] / "artifacts" / "youtube_audio_cache"


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
    youtube_info: YouTubeInfo | None = None
    subtitle_used: bool = False


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

    @staticmethod
    def _is_youtube_url(url_or_path: str) -> bool:
        """Check if the input is a YouTube URL."""
        youtube_patterns = [
            r"(?:https?:\/\/)?(?:www\.)?youtube\.com\/watch\?v=",
            r"(?:https?:\/\/)?(?:www\.)?youtu\.be\/",
            r"(?:https?:\/\/)?(?:www\.)?youtube\.com\/playlist\?list=",
        ]
        return any(re.search(pattern, str(url_or_path)) for pattern in youtube_patterns)

    def _handle_youtube(self, url: str, describe_chapters: bool = True) -> tuple[Path, YouTubeInfo, bool]:
        """Handle YouTube URL: check license, extract subtitles, download audio if needed.
        
        Returns:
            (audio_path, youtube_info, subtitle_used)
        """
        logger.info("Processing YouTube URL: %s", url)

        # Get YouTube video info
        yt_info = get_youtube_info(url)
        logger.info(
            "YouTube video info: title=%s, duration=%.1fs, license=%s (allowed=%s), "
            "has_subtitles=%s (type=%s)",
            yt_info.title,
            yt_info.duration,
            yt_info.license_info or "None",
            yt_info.license_allowed,
            yt_info.has_subtitles,
            yt_info.subtitle_type,
        )

        if not yt_info.license_allowed:
            logger.warning(
                "Video license is not CC-BY or other creative license: %s",
                yt_info.license_info or "None",
            )

        subtitle_used = False

        # If subtitles are available, we can use them instead of Whisper
        if yt_info.has_subtitles and yt_info.subtitle_content:
            logger.info("Subtitles found (%s), will skip Whisper ASR", yt_info.subtitle_type)
            subtitle_used = True
            # We don't need to download audio, segments will be created from subtitles
            # Return a dummy path
            return Path(f"youtube://{yt_info.video_id}"), yt_info, subtitle_used

        # If no subtitles, download audio
        YOUTUBE_AUDIO_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        audio_path = YOUTUBE_AUDIO_CACHE_DIR / f"{yt_info.video_id}.wav"

        if audio_path.exists():
            logger.info("Using cached YouTube audio: %s", audio_path)
        else:
            logger.info("Downloading YouTube audio to %s", audio_path)
            audio_path = download_youtube_audio(url, audio_path)

        return audio_path, yt_info, subtitle_used

    def _save_result(
        self,
        audio_path: str | Path,
        result: PipelineResult,
        *,
        describe_chapters: bool,
        duration_seconds: float,
    ) -> Path:
        if result.youtube_info:
            source_file = str(audio_path)
            artifact_stem = f"youtube_{result.youtube_info.video_id}"
        else:
            source = Path(audio_path).resolve()
            source_file = str(source)
            artifact_stem = source.stem
        timestamp = datetime.now(timezone.utc)
        
        # Include YouTube info if available
        youtube_metadata = None
        if result.youtube_info:
            youtube_metadata = {
                "video_id": result.youtube_info.video_id,
                "title": result.youtube_info.title,
                "duration": result.youtube_info.duration,
                "license_info": result.youtube_info.license_info,
                "license_allowed": result.youtube_info.license_allowed,
                "has_subtitles": result.youtube_info.has_subtitles,
                "subtitle_type": result.youtube_info.subtitle_type,
                "subtitle_used": result.subtitle_used,
            }
        
        artifact = {
            "summary": {
                "created_at": timestamp.isoformat(),
                "source_file": source_file,
                "duration_seconds": round(duration_seconds, 3),
                "generate_descriptions": describe_chapters,
                "chapter_count": len(result.chapters),
                "youtube_metadata": youtube_metadata,
                "models": {
                    "asr": {
                        "name": "faster-whisper" if not result.subtitle_used else "youtube-subtitles",
                        "model": self.config.whisper.model_size if not result.subtitle_used else None,
                        "device": self.config.whisper.device if not result.subtitle_used else None,
                        "compute_type": self.config.whisper.compute_type if not result.subtitle_used else None,
                        "language": self.config.whisper.language if not result.subtitle_used else None,
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
        output_path = SEGMENTED_ARTIFACTS_DIR / f"{artifact_stem}_{timestamp:%Y%m%dT%H%M%S%fZ}.json"
        output_path.write_text(json.dumps(artifact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        logger.info("Saved segmentation result to %s", output_path)
        return output_path

    def run(self, audio_path: str | Path, describe_chapters: bool = True) -> PipelineResult:
        logger.info("Segmentation pipeline started for %s", audio_path)
        started_at = time.perf_counter()
        
        # Handle YouTube URL
        youtube_info = None
        subtitle_used = False
        audio_for_transcription = audio_path
        
        if self._is_youtube_url(str(audio_path)):
            logger.info("Detected YouTube URL, processing...")
            audio_for_transcription, youtube_info, subtitle_used = self._handle_youtube(
                str(audio_path), describe_chapters
            )
        
        # Transcription: either from subtitles or Whisper
        if subtitle_used and youtube_info and youtube_info.subtitle_content:
            logger.info("Using extracted subtitles instead of Whisper ASR")
            segments = WhisperTranscriber.segments_from_text(
                youtube_info.subtitle_content,
                youtube_info.duration
            )
        else:
            logger.info("Using Whisper for ASR transcription")
            segments = self.transcriber.transcribe(audio_for_transcription)
        
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

        result = PipelineResult(
            chapters=chapters,
            course_summary=course_summary,
            youtube_info=youtube_info,
            subtitle_used=subtitle_used
        )
        duration_seconds = time.perf_counter() - started_at
        artifact_path = self._save_result(
            audio_path,
            result,
            describe_chapters=describe_chapters,
            duration_seconds=duration_seconds,
        )
        logger.info("Segmentation pipeline finished in %.1fs; result saved to %s", duration_seconds, artifact_path)
        return result
