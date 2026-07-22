from __future__ import annotations

from pathlib import Path
from typing import Sequence


def _to_float_list(values: Sequence[float | int]) -> list[float]:
    return [float(v) for v in values]


def _find_nearest_index(sorted_values: Sequence[float], target: float) -> int:
    if not sorted_values:
        return 0
    best_idx = 0
    best_dist = abs(float(sorted_values[0]) - float(target))
    for idx in range(1, len(sorted_values)):
        cur_dist = abs(float(sorted_values[idx]) - float(target))
        if cur_dist < best_dist:
            best_idx = idx
            best_dist = cur_dist
    return best_idx


def plot_probs_by_sentence(
    probs: Sequence[float | int],
    gt_sentence_indices: Sequence[int],
    save_path: str | Path,
    *,
    title: str | None = None,
    threshold: float | None = None,
) -> str:
    """Plot one sample's probability curve over sentence indices with GT markers."""

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required for plotting. Install it with 'pip install matplotlib'."
        ) from exc

    y = _to_float_list(probs)
    x = list(range(len(y)))

    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 4), dpi=140)
    ax.plot(x, y, color="#1f77b4", linewidth=1.8, label="Model probability")

    gt_idx = sorted({int(i) for i in gt_sentence_indices if 0 <= int(i) < len(y)})
    if gt_idx:
        gt_y = [y[i] for i in gt_idx]
        ax.scatter(gt_idx, gt_y, color="#d62728", s=28, zorder=3, label="GT boundary")
        for idx in gt_idx:
            ax.axvline(idx, color="#d62728", linestyle="--", linewidth=0.8, alpha=0.35)

    if threshold is not None:
        ax.axhline(float(threshold), color="#2ca02c", linestyle="--", linewidth=1.0, label="Threshold")

    ax.set_xlabel("Sentence index")
    ax.set_ylabel("Boundary probability")
    ax.set_ylim(0.0, 1.0)
    ax.set_xlim(0, max(0, len(y) - 1))
    ax.grid(True, linestyle=":", alpha=0.35)
    if title:
        ax.set_title(title)
    ax.legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def plot_probs_by_time(
    probs: Sequence[float | int],
    sentence_times: Sequence[float | int],
    gt_times: Sequence[float | int],
    save_path: str | Path,
    *,
    duration: float | int | None = None,
    title: str | None = None,
    threshold: float | None = None,
) -> str:
    """Plot one sample's probability curve over time with GT boundary markers."""

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required for plotting. Install it with 'pip install matplotlib'."
        ) from exc

    y = _to_float_list(probs)
    t = _to_float_list(sentence_times)
    usable_len = min(len(y), len(t))
    y = y[:usable_len]
    t = t[:usable_len]

    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 4), dpi=140)
    ax.plot(t, y, color="#1f77b4", linewidth=1.8, label="Model probability")

    gt_filtered = sorted({float(v) for v in gt_times if float(v) >= 0.0})
    if duration is not None:
        end_t = float(duration)
        gt_filtered = [v for v in gt_filtered if v <= end_t]

    if gt_filtered and t:
        gt_x = []
        gt_y = []
        for gt in gt_filtered:
            nearest_idx = _find_nearest_index(t, gt)
            gt_x.append(gt)
            gt_y.append(y[nearest_idx])

        ax.scatter(gt_x, gt_y, color="#d62728", s=28, zorder=3, label="GT boundary")
        for gt in gt_x:
            ax.axvline(gt, color="#d62728", linestyle="--", linewidth=0.8, alpha=0.35)

    if threshold is not None:
        ax.axhline(float(threshold), color="#2ca02c", linestyle="--", linewidth=1.0, label="Threshold")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Boundary probability")
    ax.set_ylim(0.0, 1.0)

    if t:
        min_t = min(t)
        max_t = max(t)
        if duration is not None:
            max_t = max(max_t, float(duration))
        if max_t <= min_t:
            max_t = min_t + 1e-6
        ax.set_xlim(min_t, max_t)

    ax.grid(True, linestyle=":", alpha=0.35)
    if title:
        ax.set_title(title)
    ax.legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return str(path)
