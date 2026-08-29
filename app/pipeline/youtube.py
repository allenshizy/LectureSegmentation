"""YouTube video handler with license checking and subtitle extraction."""

from __future__ import annotations

import logging
import re
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# Creative Commons licenses that allow commercial/derivative use
ALLOWED_LICENSES = {
    "CC BY",
    "CC-BY",
    "Creative Commons Attribution",
    "cc by",
    "cc-by",
}


@dataclass
class YouTubeInfo:
    """Information extracted from a YouTube video."""

    video_id: str
    title: str
    duration: float
    license_info: str | None
    license_allowed: bool
    has_subtitles: bool
    subtitle_type: str | None  # "auto" or "manual"
    subtitle_content: str | None


def _check_license_allowed(license_text: str | None) -> bool:
    """Check if the video license allows creative use.
    
    Only accepts pure CC-BY licenses, not CC-BY-NC, CC-BY-SA, etc.
    """
    if not license_text:
        return False
    license_text = license_text.strip().upper()
    
    # Check for CC-BY specifically (not CC-BY-NC, CC-BY-SA, etc.)
    # Match patterns like "CC-BY", "CC BY", "CC-BY 4.0", "CC-BY-4.0", etc.
    import re
    
    patterns = [
        r"CC\s*-?\s*BY(?:\s*-?\s*4\.0)?(?:\s|$|[,\)])",  # CC-BY, CC-BY-4.0 with word boundaries
        r"CREATIVE\s+COMMONS\s+ATTRIBUTION(?:\s+4\.0)?(?:\s|$|[,\)])",  # Creative Commons Attribution
    ]
    
    for pattern in patterns:
        if re.search(pattern, license_text):
            # Make sure it's not CC-BY-NC, CC-BY-SA, CC-BY-ND
            if any(x in license_text for x in ["NC", "SA", "ND"]):
                return False
            return True
    
    return False


def get_youtube_info(url: str) -> YouTubeInfo:
    """Get info about a YouTube video, including license and subtitle availability."""
    logger.info("Fetching YouTube video info: %s", url)
    started_at = time.perf_counter()

    try:
        import yt_dlp
    except ImportError:
        raise ImportError(
            "yt-dlp is required for YouTube support. Install it with: pip install yt-dlp"
        )

    ydl_opts = {
        "quiet": False,
        "no_warnings": False,
        "extract_flat": False,
        "noplaylist": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            info = ydl.extract_info(url, download=False)
        except Exception as exc:
            raise ValueError(f"Failed to fetch YouTube video info: {exc}") from exc

    video_id = info.get("id", "")
    title = info.get("title", "Unknown")
    duration = info.get("duration", 0)
    license_info = info.get("license", None)
    license_allowed = _check_license_allowed(license_info)

    subtitles = info.get("subtitles", {}) or {}
    automatic_captions = info.get("automatic_captions", {}) or {}

    # Prefer manual subtitles in English or other languages
    if subtitles:
        subtitle_type = "manual"
        subtitle_language = "en" if "en" in subtitles else next(iter(subtitles))
    elif automatic_captions:
        subtitle_type = "auto"
        subtitle_language = "en" if "en" in automatic_captions else next(iter(automatic_captions))
    else:
        subtitle_type = None
        subtitle_language = None

    subtitle_content = None
    if subtitle_type and subtitle_language:
        try:
            subtitle_content = _download_subtitle_text(url, subtitle_language, subtitle_type)
        except Exception:
            logger.warning("Could not download %s subtitles for %s", subtitle_type, video_id, exc_info=True)
    has_subtitles = bool(subtitle_content)

    logger.info(
        "YouTube info retrieved in %.1fs: video_id=%s, title=%s, duration=%.1fs, "
        "license=%s (allowed=%s), has_subtitles=%s (type=%s)",
        time.perf_counter() - started_at,
        video_id,
        title,
        duration,
        license_info or "None",
        license_allowed,
        has_subtitles,
        subtitle_type,
    )

    return YouTubeInfo(
        video_id=video_id,
        title=title,
        duration=duration,
        license_info=license_info,
        license_allowed=license_allowed,
        has_subtitles=has_subtitles,
        subtitle_type=subtitle_type,
        subtitle_content=subtitle_content,
    )


def _download_subtitle_text(url: str, language: str, subtitle_type: str) -> str | None:
    """Download one subtitle track and return its visible VTT text."""
    import yt_dlp

    with tempfile.TemporaryDirectory() as directory:
        ydl_opts = {
            "skip_download": True,
            "noplaylist": True,
            "quiet": True,
            "no_warnings": True,
            "writesubtitles": subtitle_type == "manual",
            "writeautomaticsub": subtitle_type == "auto",
            "subtitleslangs": [language],
            "subtitlesformat": "vtt/best",
            "outtmpl": str(Path(directory) / "%(id)s.%(ext)s"),
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.extract_info(url, download=True)

        subtitle_files = list(Path(directory).glob("*.vtt"))
        if not subtitle_files:
            return None
        return _parse_vtt_text(subtitle_files[0])


def _parse_vtt_text(path: Path) -> str | None:
    """Extract visible caption text from a WebVTT subtitle file."""
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    texts: list[str] = []
    previous_text = None
    for line in lines:
        text = line.strip()
        if not text or text == "WEBVTT" or text.startswith(("NOTE", "STYLE", "REGION")):
            continue
        if "-->" in text or re.fullmatch(r"\d+", text):
            continue
        text = re.sub(r"<[^>]+>", "", text)
        if text and text != previous_text:
            texts.append(text)
            previous_text = text
    return " ".join(texts) or None


def download_youtube_audio(url: str, output_path: str | Path) -> Path:
    """Download audio from YouTube video."""
    logger.info("Downloading YouTube audio: %s -> %s", url, output_path)
    started_at = time.perf_counter()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        import yt_dlp
    except ImportError:
        raise ImportError(
            "yt-dlp is required for YouTube support. Install it with: pip install yt-dlp"
        )

    ydl_opts = {
        "format": "bestaudio/best",
        "noplaylist": True,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "192",
            }
        ],
        "outtmpl": str(output_path.with_suffix(".%(ext)s")),
        "quiet": False,
        "no_warnings": False,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            ydl.download([url])
        except Exception as exc:
            raise RuntimeError(f"Failed to download YouTube audio: {exc}") from exc

    # The output should be a .wav file
    result_file = output_path.with_suffix(".wav")
    if not result_file.exists():
        # Try to find the file that was created
        parent = output_path.parent
        for f in parent.glob(output_path.stem + ".*"):
            result_file = f
            break

    if not result_file.exists():
        raise RuntimeError(f"Audio file not found after download: {result_file}")

    logger.info(
        "YouTube audio download finished in %.1fs: %s",
        time.perf_counter() - started_at,
        result_file,
    )
    return result_file
