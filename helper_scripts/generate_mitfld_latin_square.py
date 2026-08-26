"""Generate anonymized Latin-square segmentation materials from MITFLD."""

from __future__ import annotations

import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import click

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.config import OllamaConfig, STBConfig
from app.pipeline.asr import Segment
from app.pipeline.llm import OllamaClient
from app.pipeline.pipeline import Chapter, _group_into_chapters
from app.pipeline.segmentation import StbSegmenter
from kpi.datasets.mitfld import mit_srt_reader
from kpi.utils.video import get_duration

METHODS = ("stb", "equal_time", "original_chapter")
ANONYMOUS_LABELS = ("A", "B", "C")
LATIN_SQUARE = (
    (0, 1, 2),
    (1, 2, 0),
    (2, 0, 1),
)


def _load_selected_video(dataset_path: Path, video_id: str) -> tuple[list[Segment], list[float], float]:
    transcript_path = dataset_path / "transcripts" / f"{video_id}.json"
    fragments_path = dataset_path / "frags" / f"{video_id}.json"
    video_path = dataset_path / "videos" / f"{video_id}.mp4"
    missing = [str(path) for path in (transcript_path, fragments_path, video_path) if not path.is_file()]
    if missing:
        raise click.ClickException(f"Missing MITFLD files for video {video_id}: {', '.join(missing)}")

    segments = [
        Segment(float(item.start), float(item.end), str(item.text).strip())
        for item in mit_srt_reader(str(transcript_path))
        if item.text.strip()
    ]
    boundaries = [float(boundary) for boundary in json.loads(fragments_path.read_text(encoding="utf-8"))]
    duration = get_duration(str(video_path))
    return segments, boundaries, duration


def _equal_time_boundaries(segments: list[Segment], duration: float, chapter_count: int) -> list[float]:
    if chapter_count <= 1 or len(segments) <= 1:
        return []

    # Boundaries remain sentence ends so all conditions show whole transcript sentences.
    target_times = [duration * index / chapter_count for index in range(1, chapter_count)]
    boundary_indices: list[int] = []
    next_min_index = 0
    for target_time in target_times:
        while next_min_index < len(segments) - 1 and segments[next_min_index].end < target_time:
            next_min_index += 1
        if next_min_index >= len(segments) - 1:
            break
        boundary_indices.append(next_min_index)
        next_min_index += 1
    return [segments[index].end for index in boundary_indices]


def _describe_chapters(chapters: list[Chapter], llm: OllamaClient) -> str | None:
    for chapter in chapters:
        try:
            description = llm.describe_chapter(chapter.text)
            chapter.title = description["title"]
            chapter.keywords = description["keywords"]
            chapter.summary = description["summary"]
        except Exception as exc:
            chapter.description_error = str(exc)

    summaries = [chapter.summary for chapter in chapters if chapter.summary]
    if not summaries:
        return None
    try:
        return llm.summarize_course(summaries)
    except Exception:
        return None


def _serialize_chapters(chapters: list[Chapter]) -> list[dict]:
    return [
        {
            "start": chapter.start,
            "end": chapter.end,
            "title": chapter.title,
            "keywords": chapter.keywords,
            "summary": chapter.summary,
            "transcript": chapter.text,
            "description_error": chapter.description_error,
        }
        for chapter in chapters
    ]


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


@click.command()
@click.option("--dataset-path", type=click.Path(path_type=Path, exists=True, file_okay=False), required=True)
@click.option(
    "--video",
    "video_ids",
    multiple=True,
    required=True,
    help="Exactly three MITFLD video IDs, in the desired study-video order.",
)
@click.option("--output-dir", type=click.Path(path_type=Path), default="artifacts/mitfld_latin_square")
@click.option(
    "--transformer-checkpoint",
    type=click.Path(path_type=Path, exists=True),
    default=REPO_ROOT / "app" / "checkpoints" / "transformer.pt",
    show_default=True,
)
@click.option(
    "--detector-checkpoint",
    type=click.Path(path_type=Path, exists=True),
    default=REPO_ROOT / "app" / "checkpoints" / "detector.pt",
    show_default=True,
)
@click.option("--threshold", type=float, default=0.75, show_default=True)
@click.option("--local-max-window", type=int, default=1, show_default=True)
@click.option("--device", default="auto", show_default=True)
@click.option("--ollama-model", default="qwen3:4b", show_default=True)
@click.option("--ollama-host", default="127.0.0.1", show_default=True)
@click.option("--ollama-port", type=int, default=11434, show_default=True)
@click.option("--ollama-auto-start/--no-ollama-auto-start", default=True, show_default=True)
@click.option("--ollama-auto-pull/--no-ollama-auto-pull", default=True, show_default=True)
def generate_mitfld_latin_square(
    dataset_path: Path,
    video_ids: tuple[str, ...],
    output_dir: Path,
    transformer_checkpoint: Path,
    detector_checkpoint: Path,
    threshold: float,
    local_max_window: int,
    device: str,
    ollama_model: str,
    ollama_host: str,
    ollama_port: int,
    ollama_auto_start: bool,
    ollama_auto_pull: bool,
) -> None:
    """Create nine public task files plus a private method-assignment summary."""

    if len(video_ids) != 3 or len(set(video_ids)) != 3:
        raise click.BadParameter("Provide exactly three distinct --video values.", param_hint="video")
    if not 0.0 <= threshold <= 1.0:
        raise click.BadParameter("threshold must be in [0, 1].")
    if local_max_window < 0:
        raise click.BadParameter("local-max-window must be >= 0.")

    public_dir = output_dir / "materials"
    private_summary_path = output_dir / "summary_private.json"
    if output_dir.exists() and any(output_dir.iterdir()):
        raise click.ClickException(f"Output directory is not empty: {output_dir}")
    public_dir.mkdir(parents=True, exist_ok=True)

    started_at = time.perf_counter()
    stb_config = STBConfig(
        transformer_checkpoint=transformer_checkpoint,
        detector_checkpoint=detector_checkpoint,
        device=device,
        threshold=threshold,
        local_max_window=local_max_window,
    )
    segmenter = StbSegmenter(stb_config)
    ollama_config = OllamaConfig(
        model=ollama_model,
        host=ollama_host,
        port=ollama_port,
        auto_start=ollama_auto_start,
        auto_pull=ollama_auto_pull,
    )
    llm = OllamaClient(ollama_config)

    results_by_video_and_method: dict[str, dict[str, dict]] = {}
    for video_id in video_ids:
        segments, original_boundaries, duration = _load_selected_video(dataset_path, video_id)
        if not segments:
            raise click.ClickException(f"Video {video_id} has no non-empty transcript segments.")

        chapter_count = len(original_boundaries) + 1
        boundaries_by_method = {
            "stb": segmenter.predict_boundaries(segments),
            "equal_time": _equal_time_boundaries(segments, duration, chapter_count),
            "original_chapter": [float(boundary) for boundary in original_boundaries],
        }
        results_by_video_and_method[video_id] = {}
        for method, boundaries in boundaries_by_method.items():
            chapters = _group_into_chapters(segments, boundaries)
            course_summary = _describe_chapters(chapters, llm)
            results_by_video_and_method[video_id][method] = {
                "course_summary": course_summary,
                "chapters": _serialize_chapters(chapters),
            }

    created_at = datetime.now(timezone.utc).isoformat()
    assignments: list[dict] = []
    for participant_index, method_indices in enumerate(LATIN_SQUARE, start=1):
        for video_index, method_index in enumerate(method_indices):
            method = METHODS[method_index]
            anonymous_label = ANONYMOUS_LABELS[method_index]
            video_id = video_ids[video_index]
            output_path = public_dir / f"participant_{participant_index}_video_{video_index + 1}.json"
            result = results_by_video_and_method[video_id][method]
            _write_json(
                output_path,
                {
                    "task": "Select the segment that completely explains a concept.",
                    "participant": f"participant_{participant_index}",
                    "video_id": video_id,
                    "segmentation_method": anonymous_label,
                    "course_summary": result["course_summary"],
                    "chapters": result["chapters"],
                },
            )
            assignments.append(
                {
                    "participant": f"participant_{participant_index}",
                    "video_id": video_id,
                    "anonymous_label": anonymous_label,
                    "method": method,
                    "material_file": str(output_path.relative_to(output_dir)),
                }
            )

    _write_json(
        private_summary_path,
        {
            "created_at": created_at,
            "study_design": {
                "task": "Select the segment that completely explains a concept.",
                "design": "3x3 Latin square",
                "participants": 3,
                "videos": list(video_ids),
                "methods": list(METHODS),
                "anonymous_method_mapping": dict(zip(ANONYMOUS_LABELS, METHODS, strict=True)),
                "equal_time_rule": "Match the original-chapter count and align boundaries to transcript sentence ends.",
            },
            "assignments": assignments,
            "runtime": {
                "duration_seconds": round(time.perf_counter() - started_at, 3),
                "python_version": platform.python_version(),
                "platform": platform.platform(),
                "dataset_path": str(dataset_path.resolve()),
                "stb_config": {
                    "transformer_checkpoint": str(stb_config.transformer_checkpoint),
                    "detector_checkpoint": str(stb_config.detector_checkpoint),
                    "device": stb_config.device,
                    "threshold": stb_config.threshold,
                    "local_max_window": stb_config.local_max_window,
                },
                "ollama_config": {
                    "model": ollama_config.model,
                    "host": ollama_config.host,
                    "port": ollama_config.port,
                    "auto_start": ollama_config.auto_start,
                    "auto_pull": ollama_config.auto_pull,
                },
            },
        },
    )
    click.echo(f"Created 9 anonymized materials in {public_dir}")
    click.echo(f"Created private assignment summary at {private_summary_path}")


if __name__ == "__main__":
    generate_mitfld_latin_square()