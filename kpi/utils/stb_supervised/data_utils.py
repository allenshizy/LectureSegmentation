"""Data loading helpers for processed text-label datasets."""

from __future__ import annotations

import random
from typing import Any

import torch
from torch.utils.data import Dataset, Subset

from kpi.utils.text_label_dataset import (
    filter_internal_boundaries,
    map_frags_to_sentence_labels,
)


class ProcessedTextLabelDataset(Dataset):
    """Dataset wrapper for samples exported by export_text_label_dataset.py."""

    def __init__(self, samples: list[dict[str, Any]]) -> None:
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = self.samples[idx]
        sentence_ends = [float(x) for x in sample.get("sentence_ends", [])]
        duration = float(sample.get("duration", 0.0))
        frags = filter_internal_boundaries(sample.get("frags", []), duration=duration)

        # Rebuild labels from cleaned boundaries to avoid stale exported labels.
        if sentence_ends:
            labels = map_frags_to_sentence_labels(sentence_ends, frags)
            label_tensor = torch.as_tensor(labels, dtype=torch.long)
        else:
            labels = sample["label"]
            if torch.is_tensor(labels):
                label_tensor = labels.to(dtype=torch.long)
            else:
                label_tensor = torch.as_tensor(labels, dtype=torch.long)

        return {
            "text": list(sample["text"]),
            "label": label_tensor,
            "sentence_ends": sentence_ends,
            "duration": duration,
            "frags": frags,
        }


def load_processed_samples(path: str) -> list[dict[str, Any]]:
    """Load processed samples from a .pt payload."""

    payload = torch.load(path, map_location="cpu")
    if isinstance(payload, dict) and "samples" in payload:
        samples = payload["samples"]
    elif isinstance(payload, list):
        samples = payload
    else:
        raise ValueError("Unsupported processed dataset payload")

    if not isinstance(samples, list) or len(samples) == 0:
        raise ValueError("Processed dataset is empty")
    return samples


def split_processed_dataset(
    dataset: ProcessedTextLabelDataset,
    ratios: list[float],
    seed: int,
) -> tuple[Subset, Subset, Subset]:
    """Split processed dataset into train/val/test subsets."""

    indices = list(range(len(dataset)))
    random.Random(seed).shuffle(indices)

    train_end = int(ratios[0] * len(indices))
    val_end = train_end + int(ratios[1] * len(indices))

    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]

    if not train_idx or not val_idx or not test_idx:
        raise ValueError("One split is empty. Adjust split ratios or dataset size")

    return Subset(dataset, train_idx), Subset(dataset, val_idx), Subset(dataset, test_idx)


def processed_text_label_collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate processed text-label samples with timing metadata."""

    return {
        "text": [sample["text"] for sample in batch],
        "label": [sample["label"] for sample in batch],
        "lengths": torch.tensor([len(sample["text"]) for sample in batch], dtype=torch.long),
        "sentence_ends": [sample["sentence_ends"] for sample in batch],
        "duration": [sample["duration"] for sample in batch],
        "frags": [sample["frags"] for sample in batch],
    }
