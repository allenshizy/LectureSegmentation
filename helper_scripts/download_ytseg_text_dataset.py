"""Download YTSeg text data and export processed .pt files for STB pretraining.

Exports each split as a torch payload compatible with
LecturePretrainDataset.from_processed_files:
    {
        "version": 1,
        "samples": [{"text": ["sent1", "sent2", ...], ...}, ...]
    }
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import click
import torch
from tqdm import tqdm

try:
    from datasets import load_dataset
except ImportError as exc:  # pragma: no cover - environment dependent
    raise SystemExit(
        "Missing dependency: datasets. Install it first, e.g. `pip install datasets`."
    ) from exc


def _parse_splits(splits: str) -> list[str]:
    aliases = {
        "val": "validation",
        "valid": "validation",
        "dev": "validation",
    }
    items = [part.strip().lower() for part in splits.split(",") if part.strip()]
    if not items:
        raise click.BadParameter("--splits must contain at least one split name")
    normalized: list[str] = []
    for item in items:
        normalized.append(aliases.get(item, item))
    return normalized


def _to_sentence_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    if value is None:
        return []
    text = str(value).strip()
    if not text:
        return []
    return [text]


@click.command()
@click.option("--output_dir", type=str, default="processed_dataset/ytseg_text", show_default=True)
@click.option("--dataset_name", type=str, default="retkowski/ytseg", show_default=True)
@click.option("--config_name", type=str, default="text", show_default=True)
@click.option("--splits", type=str, default="train,validation,test", show_default=True)
@click.option("--transcript_field", type=str, default="text_ref", show_default=True)
@click.option("--cache_dir", type=str, default=None)
@click.option("--max_samples_per_split", type=int, default=None)
def main(
    output_dir: str,
    dataset_name: str,
    config_name: str,
    splits: str,
    transcript_field: str,
    cache_dir: str | None,
    max_samples_per_split: int | None,
) -> None:
    """Download YTSeg text subset and save split-wise processed .pt files."""

    if max_samples_per_split is not None and max_samples_per_split <= 0:
        raise click.BadParameter("--max_samples_per_split must be > 0 when provided")

    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    requested_splits = _parse_splits(splits)
    manifest: dict[str, Any] = {
        "dataset_name": dataset_name,
        "config_name": config_name,
        "transcript_field": transcript_field,
        "files": {},
    }

    for split_name in requested_splits:
        print(f"Loading split '{split_name}' from {dataset_name}:{config_name} ...")
        ds = load_dataset(
            dataset_name,
            config_name,
            split=split_name,
            cache_dir=cache_dir,
        )

        samples: list[dict[str, Any]] = []
        total_rows = len(ds)
        limit = min(total_rows, max_samples_per_split) if max_samples_per_split else total_rows

        for row_idx in tqdm(range(limit), desc=f"Processing {split_name}"):
            row = ds[row_idx]
            sentences = _to_sentence_list(row.get(transcript_field))
            if len(sentences) < 2:
                continue
            samples.append(
                {
                    "text": sentences,
                    "video_id": row.get("video_id"),
                    "channel_id": row.get("channel_id"),
                    "duration": float(row.get("duration", 0.0) or 0.0),
                }
            )

        payload = {
            "version": 1,
            "dataset": dataset_name,
            "config": config_name,
            "split": split_name,
            "transcript_field": transcript_field,
            "samples": samples,
        }

        out_path = target_dir / f"ytseg_text_{split_name}.pt"
        torch.save(payload, out_path)

        manifest["files"][split_name] = {
            "path": str(out_path),
            "rows_loaded": total_rows,
            "rows_exported": len(samples),
        }
        print(
            f"Saved {len(samples)} samples from split '{split_name}' "
            f"(source rows: {total_rows}) to {out_path}"
        )

    manifest_path = target_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as fp:
        json.dump(manifest, fp, indent=2, ensure_ascii=True)
    print(f"Saved manifest to {manifest_path}")


if __name__ == "__main__":
    main()
