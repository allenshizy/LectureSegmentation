"""Export MITFLD into a processed text-label dataset file."""

from __future__ import annotations

from pathlib import Path

import click
import torch

from kpi.datasets.mitfld import MITFLD
from kpi.utils import filter_internal_boundaries, video_to_text_and_labels


@click.command()
@click.option(
    "--dataset_path",
    required=True,
    type=str,
    help="Path to the MITFLD root directory.",
)
@click.option(
    "--output_path",
    required=True,
    type=str,
    help="Output .pt path for processed text-label dataset.",
)
def export_text_label_dataset(dataset_path: str, output_path: str) -> None:
    dataset = MITFLD(dataset_path)
    samples = []

    for idx in range(len(dataset)):
        video, frags = dataset[idx]
        text, labels = video_to_text_and_labels(video, frags)
        internal_frags = filter_internal_boundaries(frags, duration=float(video.duration))
        samples.append(
            {
                "text": list(text),
                "label": [int(x) for x in labels],
                "sentence_ends": [float(segment.end) for segment in video.srt],
                "duration": float(video.duration),
                "frags": internal_frags,
            }
        )

    payload = {"version": 1, "samples": samples}
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, target)
    print(f"Exported {len(samples)} samples to {target}")


if __name__ == "__main__":
    export_text_label_dataset()
