from __future__ import annotations

from typing import Iterable, Sequence


def labels_to_boundary_times(
    sentence_starts: Sequence[float],
    labels: Sequence[int | bool],
    *,
    include_zero: bool = False,
) -> list[float]:
    if len(sentence_starts) != len(labels):
        raise ValueError(
            "sentence_starts and labels must have the same length: "
            f"{len(sentence_starts)} != {len(labels)}"
        )

    boundary_times = [
        float(sentence_start)
        for sentence_start, label in zip(sentence_starts, labels)
        if int(label) > 0
    ]
    boundary_times = sorted(boundary_times)

    if include_zero and (not boundary_times or boundary_times[0] != 0.0):
        boundary_times = [0.0] + boundary_times

    return boundary_times


def video_labels_to_boundary_times(
    video,
    labels: Sequence[int | bool],
    *,
    include_zero: bool = False,
) -> list[float]:
    sentence_starts = [segment.start for segment in video.srt]
    return labels_to_boundary_times(
        sentence_starts,
        labels,
        include_zero=include_zero,
    )
