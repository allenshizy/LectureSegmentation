"""Prediction helpers for supervised STB boundary decoding."""

from __future__ import annotations

from typing import Sequence

import numpy as np


def local_max_boundary_indices(
    probs: Sequence[float],
    threshold: float,
    k: int,
) -> list[int]:
    """Pick boundary indices where probability exceeds threshold and is a local max in +-k."""

    if k < 0:
        raise ValueError("k must be >= 0")

    arr = np.asarray(probs, dtype=float)
    selected: list[int] = []

    for idx, cur_prob in enumerate(arr):
        if not np.isfinite(cur_prob) or float(cur_prob) <= threshold:
            continue

        left = max(0, idx - k)
        right = min(arr.size, idx + k + 1)
        if cur_prob >= float(np.max(arr[left:right])):
            selected.append(idx)

    return selected


def probs_to_boundary_times_local_max(
    probs: Sequence[float],
    sentence_ends: Sequence[float],
    threshold: float,
    k: int,
) -> list[float]:
    """Decode probability sequence into boundary times using threshold + local-max rule."""

    cur_len = min(len(probs), len(sentence_ends))
    if cur_len <= 0:
        return []

    indices = local_max_boundary_indices(
        probs=list(probs[:cur_len]),
        threshold=threshold,
        k=k,
    )
    return [float(sentence_ends[idx]) for idx in indices]
