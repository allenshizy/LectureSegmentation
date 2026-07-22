from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from torch.utils.data import Dataset as TorchDataset

if TYPE_CHECKING:
    from kpi.datasets.base_dataset import Dataset


def filter_internal_boundaries(
    boundaries: list[float | int],
    duration: float | None = None,
    tol: float = 1e-6,
) -> list[float]:
    """Return sorted unique boundaries strictly inside (0, duration)."""

    cleaned = sorted({float(x) for x in boundaries})
    if duration is None:
        return [x for x in cleaned if x > 0.0 + tol]
    max_t = float(duration)
    return [x for x in cleaned if (0.0 + tol) < x < (max_t - tol)]


def map_frags_to_sentence_labels(
    sentence_ends: list[float], frags: list[float]
) -> list[int]:
    labels = []
    last_end = -1.0
    for cur_end in sentence_ends:
        cur_label = 0
        for frag in frags:
            if last_end <= frag < cur_end:
                cur_label = 1
                break
        labels.append(cur_label)
        last_end = cur_end
    return labels


def video_to_text_and_labels(video, frags: list[float]) -> tuple[list[str], list[int]]:
    sentences = [segment.text for segment in video.srt]
    sentence_ends = [segment.end for segment in video.srt]
    internal_frags = filter_internal_boundaries(frags, duration=float(video.duration))
    labels = map_frags_to_sentence_labels(sentence_ends, internal_frags)
    return sentences, labels


class TextLabelSequenceDataset(TorchDataset):
    def __init__(self, dataset: "Dataset"):
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        video, frags = self.dataset[idx]
        text, labels = video_to_text_and_labels(video, frags)
        return {
            "text": text,
            "label": torch.tensor(labels, dtype=torch.long),
        }


def text_label_collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "text": [sample["text"] for sample in batch],
        "label": [sample["label"] for sample in batch],
        "lengths": torch.tensor(
            [len(sample["text"]) for sample in batch], dtype=torch.long
        ),
    }
