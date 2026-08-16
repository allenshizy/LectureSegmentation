"""Merge short text chunks in processed dataset payloads.

By default, chunks with word count <= 3 are merged into the previous chunk.
Useful for cleaning transcript-like chunking outputs before pretraining.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import click
import torch


def _word_count(text: str) -> int:
    return len([token for token in text.strip().split() if token])


def _is_short_chunk(text: str, min_words: int, include_equal: bool) -> bool:
    count = _word_count(text)
    if include_equal:
        return count <= min_words
    return count < min_words


def _merge_short_chunks(
    chunks: list[str],
    min_words: int,
    include_equal: bool,
) -> tuple[list[str], dict[str, int]]:
    merged: list[str] = []
    stats = {
        "short_chunks_found": 0,
        "short_chunks_merged": 0,
        "leading_short_chunks_unmerged": 0,
    }

    for chunk in chunks:
        text = str(chunk).strip()
        if not text:
            continue

        if _is_short_chunk(text, min_words=min_words, include_equal=include_equal):
            stats["short_chunks_found"] += 1
            if merged:
                merged[-1] = f"{merged[-1]} {text}".strip()
                stats["short_chunks_merged"] += 1
            else:
                # There is no previous chunk to merge into.
                merged.append(text)
                stats["leading_short_chunks_unmerged"] += 1
            continue

        merged.append(text)

    return merged, stats


def _extract_samples(payload: Any) -> tuple[Any, list[dict[str, Any]]]:
    if isinstance(payload, dict) and "samples" in payload:
        samples = payload["samples"]
        if not isinstance(samples, list):
            raise ValueError("payload['samples'] must be a list")
        return payload, samples

    if isinstance(payload, list):
        if not all(isinstance(x, dict) for x in payload):
            raise ValueError("list payload must contain dict samples")
        return None, payload

    raise ValueError("Unsupported payload format: expected dict with samples or list")


def _resolve_input_paths(input_paths: tuple[str, ...], input_dir: str, glob_pattern: str) -> list[Path]:
    if input_paths:
        return [Path(p) for p in input_paths]

    resolved = sorted(Path(input_dir).glob(glob_pattern))
    if not resolved:
        raise click.BadParameter(
            f"No files matched {glob_pattern!r} under {input_dir!r}. "
            "Pass --input_path explicitly or adjust --input_dir/--glob_pattern."
        )
    return resolved


@click.command()
@click.option(
    "--input_path",
    "input_paths",
    multiple=True,
    type=str,
    help="Path to a processed .pt file. Repeat this option for multiple files.",
)
@click.option(
    "--input_dir",
    type=str,
    default="processed_dataset/openphi_textbook",
    show_default=True,
    help="Directory used when --input_path is not provided.",
)
@click.option(
    "--glob_pattern",
    type=str,
    default="openphi_textbooks_*.pt",
    show_default=True,
    help="Glob pattern used under --input_dir when --input_path is not provided.",
)
@click.option(
    "--output_dir",
    type=str,
    default="processed_dataset/openphi_textbook_merged_short",
    show_default=True,
)
@click.option("--min_words", type=int, default=3, show_default=True)
@click.option(
    "--include_equal/--exclude_equal",
    default=True,
    show_default=True,
    help="If enabled, chunks with word count <= min_words are merged.",
)
def main(
    input_paths: tuple[str, ...],
    input_dir: str,
    glob_pattern: str,
    output_dir: str,
    min_words: int,
    include_equal: bool,
) -> None:
    """Merge short chunks into previous chunk and write updated .pt files."""

    if min_words < 1:
        raise click.BadParameter("--min_words must be >= 1")

    source_paths = _resolve_input_paths(input_paths, input_dir=input_dir, glob_pattern=glob_pattern)
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    report: dict[str, Any] = {
        "input_files": [str(p) for p in source_paths],
        "output_dir": str(target_dir),
        "min_words": min_words,
        "include_equal": include_equal,
        "files": {},
    }

    for source_path in source_paths:
        payload = torch.load(source_path, map_location="cpu")
        payload_container, samples = _extract_samples(payload)

        new_samples: list[dict[str, Any]] = []
        total_before_chunks = 0
        total_after_chunks = 0
        total_short_found = 0
        total_short_merged = 0
        total_leading_unmerged = 0

        for sample in samples:
            text_chunks = [str(x) for x in sample.get("text", [])]
            total_before_chunks += len(text_chunks)

            merged_chunks, stats = _merge_short_chunks(
                chunks=text_chunks,
                min_words=min_words,
                include_equal=include_equal,
            )

            total_short_found += stats["short_chunks_found"]
            total_short_merged += stats["short_chunks_merged"]
            total_leading_unmerged += stats["leading_short_chunks_unmerged"]
            total_after_chunks += len(merged_chunks)

            new_sample = dict(sample)
            new_sample["text"] = merged_chunks
            new_samples.append(new_sample)

        if payload_container is None:
            new_payload: Any = new_samples
        else:
            new_payload = dict(payload_container)
            new_payload["samples"] = new_samples
            new_payload["short_chunk_merge"] = {
                "min_words": min_words,
                "include_equal": include_equal,
                "mode": "merge_to_previous",
            }

        out_path = target_dir / source_path.name
        torch.save(new_payload, out_path)

        report["files"][source_path.name] = {
            "input_path": str(source_path),
            "output_path": str(out_path),
            "num_samples": len(samples),
            "chunks_before": total_before_chunks,
            "chunks_after": total_after_chunks,
            "short_chunks_found": total_short_found,
            "short_chunks_merged": total_short_merged,
            "leading_short_chunks_unmerged": total_leading_unmerged,
        }

        print(
            f"{source_path.name}: samples={len(samples)} "
            f"chunks {total_before_chunks}->{total_after_chunks} "
            f"short_found={total_short_found} merged={total_short_merged} "
            f"leading_unmerged={total_leading_unmerged}"
        )

    report_path = target_dir / "merge_short_chunks_report.json"
    with report_path.open("w", encoding="utf-8") as fp:
        json.dump(report, fp, indent=2, ensure_ascii=True)
    print(f"Saved report to {report_path}")


if __name__ == "__main__":
    main()
