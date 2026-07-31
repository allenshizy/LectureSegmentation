"""Build chapter-level processed datasets from open-phi/textbooks.

This script converts each textbook into multiple samples where each sample is one
chapter-level section (split by level-2 markdown headings), then chunks chapter
content by comma to mimic transcript-like speaking style.

Exports three processed .pt files with explicit split metadata:
    openphi_textbooks_train.pt
    openphi_textbooks_validation.pt
    openphi_textbooks_test.pt
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Any

import click
import torch
from tqdm import tqdm

from kpi.utils.stb_supervised import validate_split_ratios

try:
    from datasets import load_dataset
except ImportError as exc:  # pragma: no cover - environment dependent
    raise SystemExit(
        "Missing dependency: datasets. Install it first, e.g. `uv add datasets`."
    ) from exc


_H2_PATTERN = re.compile(r"(?m)^##\s+(.+?)\s*$")
_LINK_PATTERN = re.compile(r"\[([^\]]+)\]\([^\)]+\)")
_CODE_BLOCK_PATTERN = re.compile(r"```.*?```", flags=re.DOTALL)
_WHITESPACE_PATTERN = re.compile(r"\s+")


def _chunk_by_comma(text: str) -> list[str]:
    parts = re.split(r"[,，]+", text)
    chunks = [part.strip() for part in parts if part and part.strip()]
    return chunks


def _clean_markdown_body(text: str) -> str:
    # Drop fenced code blocks first.
    text = _CODE_BLOCK_PATTERN.sub(" ", text)
    lines = text.splitlines()
    cleaned_lines: list[str] = []

    for raw in lines:
        line = raw.strip()
        if not line:
            continue

        # Remove heading/list/quote markdown markers.
        line = re.sub(r"^#{1,6}\s+", "", line)
        line = re.sub(r"^[-*+]\s+", "", line)
        line = re.sub(r"^>\s+", "", line)

        # Replace markdown links with anchor text.
        line = _LINK_PATTERN.sub(r"\1", line)

        # Skip repeated AI disclaimer noise blocks.
        upper_line = line.upper()
        if "NOTE - THIS TEXTBOOK WAS AI GENERATED" in upper_line:
            continue
        if "THIS TEXTBOOK WAS GENERATED USING AI TECHNIQUES" in upper_line:
            continue

        cleaned_lines.append(line)

    merged = " ".join(cleaned_lines)
    merged = _WHITESPACE_PATTERN.sub(" ", merged).strip()
    return merged


def _is_chapter_title(title: str) -> bool:
    normalized = title.strip().lower()
    return "chapter" in normalized


def _extract_chapter_sections(markdown: str, include_non_chapter_h2: bool) -> list[tuple[str, str]]:
    """Return list of (title, body_text) for chapter-level sections from H2."""

    matches = list(_H2_PATTERN.finditer(markdown))
    if not matches:
        return []

    sections: list[tuple[str, str]] = []
    for idx, match in enumerate(matches):
        title = match.group(1).strip()
        if (not include_non_chapter_h2) and (not _is_chapter_title(title)):
            continue

        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(markdown)
        body = markdown[start:end].strip()
        if not body:
            continue
        sections.append((title, body))
    return sections


def _split_indices(total_size: int, ratios: tuple[float, float, float], seed: int) -> tuple[list[int], list[int], list[int]]:
    indices = list(range(total_size))
    random.Random(seed).shuffle(indices)

    train_end = int(ratios[0] * total_size)
    val_end = train_end + int(ratios[1] * total_size)

    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]
    return train_idx, val_idx, test_idx


def _build_payload(
    samples: list[dict[str, Any]],
    selected_indices: list[int],
    split_name: str,
    seed: int,
    ratios: tuple[float, float, float],
    dataset_name: str,
    config_name: str,
) -> dict[str, Any]:
    return {
        "version": 1,
        "dataset": dataset_name,
        "config": config_name,
        "seed": seed,
        "ratios": {
            "train": ratios[0],
            "validation": ratios[1],
            "test": ratios[2],
        },
        "split": split_name,
        "samples": [samples[idx] for idx in selected_indices],
    }


@click.command()
@click.option("--output_dir", type=str, default="processed_dataset/openphi_textbooks", show_default=True)
@click.option("--dataset_name", type=str, default="open-phi/textbooks", show_default=True)
@click.option("--config_name", type=str, default="default", show_default=True)
@click.option("--source_split", type=str, default="train", show_default=True)
@click.option("--seed", type=int, default=2024, show_default=True)
@click.option("--train_ratio", type=float, default=0.8, show_default=True)
@click.option("--val_ratio", type=float, default=0.1, show_default=True)
@click.option("--test_ratio", type=float, default=0.1, show_default=True)
@click.option(
    "--include_non_chapter_h2/--no-include_non_chapter_h2",
    default=False,
    show_default=True,
    help="When disabled, only H2 titles containing 'chapter' are used as samples.",
)
@click.option("--min_chunks", type=int, default=2, show_default=True)
@click.option("--max_books", type=int, default=None, help="Optional cap on number of textbooks to process.")
@click.option("--cache_dir", type=str, default=None)
def main(
    output_dir: str,
    dataset_name: str,
    config_name: str,
    source_split: str,
    seed: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    include_non_chapter_h2: bool,
    min_chunks: int,
    max_books: int | None,
    cache_dir: str | None,
) -> None:
    """Export open-phi/textbooks as chapter-level processed .pt split files."""

    if min_chunks < 2:
        raise click.BadParameter("--min_chunks must be >= 2 for pretrain compatibility")
    if max_books is not None and max_books <= 0:
        raise click.BadParameter("--max_books must be > 0 when provided")

    validate_split_ratios(train_ratio, val_ratio, test_ratio)
    ratios = (train_ratio, val_ratio, test_ratio)

    print(f"Loading {dataset_name}:{config_name}:{source_split} ...")
    ds = load_dataset(
        dataset_name,
        config_name,
        split=source_split,
        cache_dir=cache_dir,
    )

    total_rows = len(ds)
    limit = min(total_rows, max_books) if max_books else total_rows

    samples: list[dict[str, Any]] = []
    chapter_total = 0

    for row_idx in tqdm(range(limit), desc="Extracting textbook chapters"):
        row = ds[row_idx]
        markdown = str(row.get("markdown") or "").strip()
        if not markdown:
            continue

        chapter_sections = _extract_chapter_sections(
            markdown=markdown,
            include_non_chapter_h2=include_non_chapter_h2,
        )
        if not chapter_sections:
            continue

        for chapter_idx, (chapter_title, chapter_body) in enumerate(chapter_sections):
            cleaned = _clean_markdown_body(chapter_body)
            if not cleaned:
                continue
            chunks = _chunk_by_comma(cleaned)
            if len(chunks) < min_chunks:
                continue

            sample_id = f"book_{row_idx}_chapter_{chapter_idx}"
            samples.append(
                {
                    "text": chunks,
                    "sample_id": sample_id,
                    "book_index": int(row_idx),
                    "chapter_index": int(chapter_idx),
                    "chapter_title": chapter_title,
                    "topic": row.get("topic"),
                    "field": row.get("field"),
                    "subfield": row.get("subfield"),
                    "source_split": source_split,
                }
            )
            chapter_total += 1

    if len(samples) < 3:
        raise click.ClickException(
            f"Need at least 3 chapter samples to split, got {len(samples)}. "
            "Try increasing --max_books or loosening chapter filters."
        )

    train_idx, val_idx, test_idx = _split_indices(total_size=len(samples), ratios=ratios, seed=seed)
    if not train_idx or not val_idx or not test_idx:
        raise click.ClickException(
            "One split became empty. Adjust ratios or provide more chapter samples."
        )

    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    split_map = {
        "train": train_idx,
        "validation": val_idx,
        "test": test_idx,
    }

    manifest: dict[str, Any] = {
        "dataset_name": dataset_name,
        "config_name": config_name,
        "source_split": source_split,
        "seed": seed,
        "ratios": {
            "train": train_ratio,
            "validation": val_ratio,
            "test": test_ratio,
        },
        "processed_books": limit,
        "source_rows_total": total_rows,
        "chapter_samples_total": chapter_total,
        "files": {},
    }

    for split_name, split_indices in split_map.items():
        payload = _build_payload(
            samples=samples,
            selected_indices=split_indices,
            split_name=split_name,
            seed=seed,
            ratios=ratios,
            dataset_name=dataset_name,
            config_name=config_name,
        )
        out_path = target_dir / f"openphi_textbooks_{split_name}.pt"
        torch.save(payload, out_path)
        manifest["files"][split_name] = {
            "path": str(out_path),
            "num_samples": len(split_indices),
        }
        print(f"Saved {split_name}: {len(split_indices)} chapter samples -> {out_path}")

    manifest_path = target_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as fp:
        json.dump(manifest, fp, indent=2, ensure_ascii=True)
    print(f"Saved manifest to {manifest_path}")


if __name__ == "__main__":
    main()
