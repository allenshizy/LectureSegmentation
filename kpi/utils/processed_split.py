"""Helpers for routing processed dataset files into pre-split buckets."""

from __future__ import annotations

from pathlib import Path

import torch


def infer_processed_split(path: str) -> str:
    """Infer canonical split name: train, validation, or test.

    Priority:
    1) payload["split"] from the .pt file when available
    2) filename heuristics
    3) default to train
    """

    split_value = ""
    try:
        payload = torch.load(path, map_location="cpu")
        if isinstance(payload, dict):
            split_value = str(payload.get("split", "")).strip().lower()
    except Exception:
        split_value = ""

    canonical = _canonicalize_split_name(split_value)
    if canonical is not None:
        return canonical

    name = Path(path).name.lower()
    if "validation" in name or "_val" in name or "-val" in name or "_dev" in name:
        return "validation"
    if "test" in name:
        return "test"
    return "train"


def partition_processed_paths(paths: list[str]) -> tuple[list[str], list[str], list[str]]:
    """Split paths into train, validation, and test buckets."""

    train_paths: list[str] = []
    val_paths: list[str] = []
    test_paths: list[str] = []
    for path in paths:
        split = infer_processed_split(path)
        if split == "validation":
            val_paths.append(path)
        elif split == "test":
            test_paths.append(path)
        else:
            train_paths.append(path)
    return train_paths, val_paths, test_paths


def _canonicalize_split_name(value: str) -> str | None:
    if not value:
        return None
    if value in {"train"}:
        return "train"
    if value in {"validation", "val", "valid", "dev"}:
        return "validation"
    if value in {"test"}:
        return "test"
    return None
