"""Compute adjacent-boundary interval statistics for a dataset."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import click
import matplotlib.pyplot as plt
import numpy as np

from kpi.datasets.mitfld import MITFLD
from kpi.utils.stb_supervised import load_processed_samples


@dataclass
class SampleInfo:
    sentence_ends: list[float]
    boundaries: list[float]


def _infer_duration(sentence_ends: list[float], boundaries: list[float], duration: float) -> float:
    if duration > 0.0:
        return float(duration)
    max_sentence_end = max(sentence_ends) if sentence_ends else 0.0
    max_boundary = max(boundaries) if boundaries else 0.0
    return float(max(max_sentence_end, max_boundary))


def _canonical_boundaries(boundaries: list[float], duration: float) -> list[float]:
    if duration <= 0.0:
        return [0.0]
    interior = sorted({float(x) for x in boundaries if 0.0 < float(x) < duration})
    return [0.0, *interior, float(duration)]


def _collect_from_raw_dataset(dataset_path: str) -> list[SampleInfo]:
    dataset = MITFLD(dataset_path)
    samples: list[SampleInfo] = []
    for idx in range(len(dataset)):
        video, frags = dataset[idx]
        sentence_ends = sorted(float(seg.end) for seg in video.srt)
        duration = _infer_duration(sentence_ends, [float(x) for x in frags], float(video.duration))
        boundaries = _canonical_boundaries([float(x) for x in frags], duration)
        samples.append(SampleInfo(sentence_ends=sentence_ends, boundaries=boundaries))
    return samples


def _collect_from_processed_dataset(processed_dataset_path: str) -> list[SampleInfo]:
    raw_samples = load_processed_samples(processed_dataset_path)
    samples: list[SampleInfo] = []
    for sample in raw_samples:
        sentence_ends = sorted(float(x) for x in sample.get("sentence_ends", []))
        frags = [float(x) for x in sample.get("frags", [])]
        duration = _infer_duration(sentence_ends, frags, float(sample.get("duration", 0.0)))
        boundaries = _canonical_boundaries(frags, duration)
        samples.append(SampleInfo(sentence_ends=sentence_ends, boundaries=boundaries))
    return samples


def _compute_interval_stats(samples: list[SampleInfo]) -> tuple[np.ndarray, np.ndarray]:
    sentence_counts: list[int] = []
    time_durations: list[float] = []

    for sample in samples:
        boundaries = sample.boundaries
        if len(boundaries) < 2:
            continue
        sentence_ends = np.asarray(sample.sentence_ends, dtype=float)
        for left, right in zip(boundaries[:-1], boundaries[1:]):
            if right <= left:
                continue
            left_idx = int(np.searchsorted(sentence_ends, left, side="right"))
            right_idx = int(np.searchsorted(sentence_ends, right, side="right"))
            sentence_counts.append(max(0, right_idx - left_idx))
            time_durations.append(float(right - left))

    return np.asarray(sentence_counts, dtype=float), np.asarray(time_durations, dtype=float)


def _print_summary(name: str, values: np.ndarray) -> None:
    if values.size == 0:
        click.echo(f"{name}: no valid intervals found.")
        return
    click.echo(f"{name} summary:")
    click.echo(f"  count:   {values.size}")
    click.echo(f"  median:  {float(np.median(values)):.4f}")
    click.echo(f"  min:     {float(np.min(values)):.4f}")
    click.echo(f"  max:     {float(np.max(values)):.4f}")
    click.echo(f"  average: {float(np.mean(values)):.4f}")


def _print_histogram(name: str, values: np.ndarray, bins: int, integer_mode: bool) -> None:
    if values.size == 0:
        click.echo(f"{name} histogram: no data.")
        return

    click.echo(f"{name} histogram:")

    if integer_mode:
        int_values = values.astype(int)
        uniq = np.unique(int_values)
        if uniq.size <= 60 and (int(np.max(int_values)) - int(np.min(int_values))) <= 200:
            for val in range(int(np.min(int_values)), int(np.max(int_values)) + 1):
                count = int(np.sum(int_values == val))
                click.echo(f"  [{val:>3d}] -> {count}")
            return

    hist, edges = np.histogram(values, bins=bins)
    for i, count in enumerate(hist):
        left = edges[i]
        right = edges[i + 1]
        click.echo(f"  [{left:.4f}, {right:.4f}) -> {int(count)}")


def _filter_outliers(
    values: np.ndarray,
    method: str,
    iqr_k: float,
    lower_pct: float,
    upper_pct: float,
) -> tuple[np.ndarray, tuple[float, float] | None]:
    if values.size == 0 or method == "none":
        return values, None

    if method == "iqr":
        q1 = float(np.percentile(values, 25))
        q3 = float(np.percentile(values, 75))
        iqr = q3 - q1
        lower = q1 - iqr_k * iqr
        upper = q3 + iqr_k * iqr
    elif method == "percentile":
        lower = float(np.percentile(values, lower_pct))
        upper = float(np.percentile(values, upper_pct))
    else:
        raise ValueError(f"Unsupported outlier method: {method}")

    keep = (values >= lower) & (values <= upper)
    return values[keep], (lower, upper)


def _print_filter_report(name: str, original: np.ndarray, filtered: np.ndarray, bounds: tuple[float, float] | None) -> None:
    if bounds is None:
        click.echo(f"{name} outlier filter: disabled")
        return
    lower, upper = bounds
    removed = int(original.size - filtered.size)
    removed_ratio = (removed / original.size * 100.0) if original.size > 0 else 0.0
    click.echo(
        f"{name} outlier filter: keep [{lower:.4f}, {upper:.4f}], removed {removed}/{original.size} ({removed_ratio:.2f}%)"
    )


def _summary_values(values: np.ndarray) -> dict[str, float | int]:
    if values.size == 0:
        return {
            "count": 0,
            "median": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
            "average": float("nan"),
        }
    return {
        "count": int(values.size),
        "median": float(np.median(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "average": float(np.mean(values)),
    }


def _build_stats_text(
    summary: dict[str, float | int],
    method: str,
    bounds: tuple[float, float] | None,
    original_count: int,
) -> str:
    if summary["count"] == 0:
        return "No valid intervals"

    lines = [
        f"count={summary['count']}",
        f"median={float(summary['median']):.4f}",
        f"min={float(summary['min']):.4f}",
        f"max={float(summary['max']):.4f}",
        f"average={float(summary['average']):.4f}",
    ]

    if method == "none" or bounds is None:
        lines.append("filter=none")
    else:
        lower, upper = bounds
        removed = max(0, original_count - int(summary["count"]))
        removed_ratio = (removed / original_count * 100.0) if original_count > 0 else 0.0
        lines.append(f"filter={method}")
        lines.append(f"keep=[{lower:.4f}, {upper:.4f}]")
        lines.append(f"removed={removed}/{original_count} ({removed_ratio:.2f}%)")

    return "\n".join(lines)


def _plot_histograms(
    sentence_raw_values: np.ndarray,
    duration_raw_values: np.ndarray,
    sentence_filtered_values: np.ndarray,
    duration_filtered_values: np.ndarray,
    bins: int,
    outlier_method: str,
    sentence_bounds: tuple[float, float] | None,
    duration_bounds: tuple[float, float] | None,
    sentence_original_count: int,
    duration_original_count: int,
    save_plot_path: str,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), constrained_layout=True)

    sentence_raw_summary = _summary_values(sentence_raw_values)
    duration_raw_summary = _summary_values(duration_raw_values)
    sentence_filtered_summary = _summary_values(sentence_filtered_values)
    duration_filtered_summary = _summary_values(duration_filtered_values)

    ax_sentence_raw = axes[0, 0]
    if sentence_raw_values.size > 0:
        ax_sentence_raw.hist(sentence_raw_values, bins=bins, color="#1f77b4", alpha=0.85, edgecolor="black")
    ax_sentence_raw.set_title("Sentences Between Adjacent Boundaries (Raw)")
    ax_sentence_raw.set_xlabel("Sentence count")
    ax_sentence_raw.set_ylabel("Frequency")
    sentence_raw_text = _build_stats_text(
        sentence_raw_summary,
        method="none",
        bounds=None,
        original_count=sentence_original_count,
    )
    ax_sentence_raw.text(
        0.98,
        0.98,
        sentence_raw_text,
        transform=ax_sentence_raw.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "white", "alpha": 0.9},
    )

    ax_duration_raw = axes[0, 1]
    if duration_raw_values.size > 0:
        ax_duration_raw.hist(duration_raw_values, bins=bins, color="#ff7f0e", alpha=0.85, edgecolor="black")
    ax_duration_raw.set_title("Duration Between Adjacent Boundaries (Raw)")
    ax_duration_raw.set_xlabel("Duration (seconds)")
    ax_duration_raw.set_ylabel("Frequency")
    duration_raw_text = _build_stats_text(
        duration_raw_summary,
        method="none",
        bounds=None,
        original_count=duration_original_count,
    )
    ax_duration_raw.text(
        0.98,
        0.98,
        duration_raw_text,
        transform=ax_duration_raw.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "white", "alpha": 0.9},
    )

    ax_sentence_filtered = axes[1, 0]
    if sentence_filtered_values.size > 0:
        ax_sentence_filtered.hist(
            sentence_filtered_values,
            bins=bins,
            color="#2ca02c",
            alpha=0.85,
            edgecolor="black",
        )
    ax_sentence_filtered.set_title("Sentences Between Adjacent Boundaries (Filtered)")
    ax_sentence_filtered.set_xlabel("Sentence count")
    ax_sentence_filtered.set_ylabel("Frequency")
    sentence_filtered_text = _build_stats_text(
        sentence_filtered_summary,
        method=outlier_method,
        bounds=sentence_bounds,
        original_count=sentence_original_count,
    )
    ax_sentence_filtered.text(
        0.98,
        0.98,
        sentence_filtered_text,
        transform=ax_sentence_filtered.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "white", "alpha": 0.9},
    )

    ax_duration_filtered = axes[1, 1]
    if duration_filtered_values.size > 0:
        ax_duration_filtered.hist(
            duration_filtered_values,
            bins=bins,
            color="#d62728",
            alpha=0.85,
            edgecolor="black",
        )
    ax_duration_filtered.set_title("Duration Between Adjacent Boundaries (Filtered)")
    ax_duration_filtered.set_xlabel("Duration (seconds)")
    ax_duration_filtered.set_ylabel("Frequency")
    duration_filtered_text = _build_stats_text(
        duration_filtered_summary,
        method=outlier_method,
        bounds=duration_bounds,
        original_count=duration_original_count,
    )
    ax_duration_filtered.text(
        0.98,
        0.98,
        duration_filtered_text,
        transform=ax_duration_filtered.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "white", "alpha": 0.9},
    )

    fig.suptitle("Boundary Interval Statistics (Raw vs Filtered)", fontsize=14)
    output_path = Path(save_plot_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


@click.command()
@click.option("--dataset_path", type=str, default=None, help="Path to MITFLD root directory.")
@click.option(
    "--processed_dataset_path",
    type=str,
    default=None,
    help="Path to processed .pt dataset created by export_text_label_dataset.py.",
)
@click.option("--bins", type=int, default=10, show_default=True, help="Histogram bins.")
@click.option(
    "--outlier_method",
    type=click.Choice(["none", "iqr", "percentile"], case_sensitive=False),
    default="none",
    show_default=True,
    help="Outlier filtering method applied before summary/histogram.",
)
@click.option(
    "--iqr_k",
    type=float,
    default=1.5,
    show_default=True,
    help="IQR multiplier when --outlier_method iqr.",
)
@click.option(
    "--lower_pct",
    type=float,
    default=1.0,
    show_default=True,
    help="Lower percentile when --outlier_method percentile.",
)
@click.option(
    "--upper_pct",
    type=float,
    default=99.0,
    show_default=True,
    help="Upper percentile when --outlier_method percentile.",
)
@click.option(
    "--save_plot_path",
    type=str,
    default="artifacts/boundary_interval_stats.png",
    show_default=True,
    help="Output path for histogram image.",
)
def main(
    dataset_path: str | None,
    processed_dataset_path: str | None,
    bins: int,
    outlier_method: str,
    iqr_k: float,
    lower_pct: float,
    upper_pct: float,
    save_plot_path: str,
) -> None:
    """Print boundary-interval sentence-count and duration distributions."""
    if bool(dataset_path) == bool(processed_dataset_path):
        raise click.BadParameter("Provide exactly one of --dataset_path or --processed_dataset_path")
    if bins <= 0:
        raise click.BadParameter("--bins must be > 0")
    if iqr_k <= 0:
        raise click.BadParameter("--iqr_k must be > 0")
    if not (0.0 <= lower_pct < upper_pct <= 100.0):
        raise click.BadParameter("Require 0 <= lower_pct < upper_pct <= 100")

    if dataset_path is not None:
        samples = _collect_from_raw_dataset(dataset_path)
    else:
        samples = _collect_from_processed_dataset(processed_dataset_path)

    sentence_counts, time_durations = _compute_interval_stats(samples)
    sentence_counts_filtered, sentence_bounds = _filter_outliers(
        sentence_counts,
        method=outlier_method,
        iqr_k=iqr_k,
        lower_pct=lower_pct,
        upper_pct=upper_pct,
    )
    time_durations_filtered, duration_bounds = _filter_outliers(
        time_durations,
        method=outlier_method,
        iqr_k=iqr_k,
        lower_pct=lower_pct,
        upper_pct=upper_pct,
    )

    click.echo(f"Loaded samples: {len(samples)}")
    click.echo("")

    _print_filter_report(
        "Sentences between adjacent boundaries",
        sentence_counts,
        sentence_counts_filtered,
        sentence_bounds,
    )
    _print_filter_report(
        "Duration between adjacent boundaries (seconds)",
        time_durations,
        time_durations_filtered,
        duration_bounds,
    )
    click.echo("")

    click.echo("[Raw] Sentences between adjacent boundaries")
    _print_summary("Sentences between adjacent boundaries", sentence_counts)
    _print_histogram(
        "Sentences between adjacent boundaries",
        sentence_counts,
        bins=bins,
        integer_mode=True,
    )
    click.echo("")

    click.echo("[Raw] Duration between adjacent boundaries (seconds)")
    _print_summary("Duration between adjacent boundaries (seconds)", time_durations)
    _print_histogram(
        "Duration between adjacent boundaries (seconds)",
        time_durations,
        bins=bins,
        integer_mode=False,
    )
    click.echo("")

    click.echo("[Filtered] Sentences between adjacent boundaries")
    _print_summary("Sentences between adjacent boundaries", sentence_counts_filtered)
    _print_histogram(
        "Sentences between adjacent boundaries",
        sentence_counts_filtered,
        bins=bins,
        integer_mode=True,
    )
    click.echo("")

    click.echo("[Filtered] Duration between adjacent boundaries (seconds)")
    _print_summary("Duration between adjacent boundaries (seconds)", time_durations_filtered)
    _print_histogram(
        "Duration between adjacent boundaries (seconds)",
        time_durations_filtered,
        bins=bins,
        integer_mode=False,
    )

    _plot_histograms(
        sentence_raw_values=sentence_counts,
        duration_raw_values=time_durations,
        sentence_filtered_values=sentence_counts_filtered,
        duration_filtered_values=time_durations_filtered,
        bins=bins,
        outlier_method=outlier_method,
        sentence_bounds=sentence_bounds,
        duration_bounds=duration_bounds,
        sentence_original_count=int(sentence_counts.size),
        duration_original_count=int(time_durations.size),
        save_plot_path=save_plot_path,
    )
    click.echo("")
    click.echo(f"Saved histogram image to: {save_plot_path}")


if __name__ == "__main__":
    main()
