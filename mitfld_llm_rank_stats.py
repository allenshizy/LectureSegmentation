from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import click
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

METRIC_KEYS = ["SC", "TT", "TC", "TP", "GC", "BN", "MB"]
ANNOTATORS = ["A", "B", "C"]
RANK_LABELS = ["rank1", "rank2", "rank3"]
METRIC_NAMES = {
    "SC": "Semantic coherence",
    "TT": "Topic transition",
    "TC": "Topic completeness",
    "TP": "Topic purity",
    "GC": "Granularity consistency",
    "BN": "Boundary necessity",
    "MB": "Missing boundary",
}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _position_map(order: str) -> dict[str, int]:
    left, mid, right = [part.strip() for part in order.split(">")]
    return {left: 1, mid: 2, right: 3}


def _extract_valid_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    valid_rows = [row for row in rows if bool(row.get("valid"))]
    return valid_rows


def _count_frequency(valid_rows: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, int]]]:
    counts: dict[str, dict[str, dict[str, int]]] = {
        metric: {
            annotator: {rank: 0 for rank in RANK_LABELS}
            for annotator in ANNOTATORS
        }
        for metric in METRIC_KEYS
    }

    for row in valid_rows:
        parsed = row.get("parsed", {})
        for metric in METRIC_KEYS:
            order = str(parsed.get(metric, ""))
            if not order:
                continue
            pos = _position_map(order)
            for annotator in ANNOTATORS:
                rank = pos.get(annotator)
                if rank == 1:
                    counts[metric][annotator]["rank1"] += 1
                elif rank == 2:
                    counts[metric][annotator]["rank2"] += 1
                elif rank == 3:
                    counts[metric][annotator]["rank3"] += 1

    return counts


def _write_summary_long(path: Path, counts: dict[str, dict[str, dict[str, int]]], valid_n: int) -> None:
    fieldnames = ["metric", "annotator", "rank", "count", "frequency"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for metric in METRIC_KEYS:
            for annotator in ANNOTATORS:
                for rank in RANK_LABELS:
                    count = counts[metric][annotator][rank]
                    freq = (count / valid_n) if valid_n > 0 else 0.0
                    writer.writerow(
                        {
                            "metric": metric,
                            "annotator": annotator,
                            "rank": rank,
                            "count": count,
                            "frequency": f"{freq:.6f}",
                        }
                    )


def _write_summary_wide(path: Path, counts: dict[str, dict[str, dict[str, int]]], valid_n: int) -> None:
    fieldnames = ["metric"]
    for annotator in ANNOTATORS:
        fieldnames.extend(
            [
                f"{annotator}_rank1_count",
                f"{annotator}_rank1_freq",
                f"{annotator}_rank2_count",
                f"{annotator}_rank2_freq",
                f"{annotator}_rank3_count",
                f"{annotator}_rank3_freq",
            ]
        )

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for metric in METRIC_KEYS:
            row: dict[str, Any] = {"metric": metric}
            for annotator in ANNOTATORS:
                c1 = counts[metric][annotator]["rank1"]
                c2 = counts[metric][annotator]["rank2"]
                c3 = counts[metric][annotator]["rank3"]
                row[f"{annotator}_rank1_count"] = c1
                row[f"{annotator}_rank1_freq"] = f"{(c1 / valid_n) if valid_n > 0 else 0.0:.6f}"
                row[f"{annotator}_rank2_count"] = c2
                row[f"{annotator}_rank2_freq"] = f"{(c2 / valid_n) if valid_n > 0 else 0.0:.6f}"
                row[f"{annotator}_rank3_count"] = c3
                row[f"{annotator}_rank3_freq"] = f"{(c3 / valid_n) if valid_n > 0 else 0.0:.6f}"
            writer.writerow(row)


def _write_frequency_plot(
    path: Path,
    counts: dict[str, dict[str, dict[str, int]]],
    valid_n: int,
) -> None:
    frequencies = {
        rank: [
            [
                (counts[metric][annotator][rank] / valid_n) if valid_n > 0 else 0.0
                for annotator in ANNOTATORS
            ]
            for metric in METRIC_KEYS
        ]
        for rank in RANK_LABELS
    }

    figure, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
    x_positions = list(range(len(METRIC_KEYS)))
    bar_width = 0.24
    colors = {"A": "#2563eb", "B": "#ea580c", "C": "#16a34a"}

    for axis, rank in zip(axes, RANK_LABELS):
        for annotator_idx, annotator in enumerate(ANNOTATORS):
            values = [row[annotator_idx] for row in frequencies[rank]]
            offsets = [x + (annotator_idx - 1) * bar_width for x in x_positions]
            bars = axis.bar(
                offsets,
                values,
                width=bar_width,
                label=annotator,
                color=colors[annotator],
            )
            axis.bar_label(
                bars,
                labels=[f"{value:.0%}" if value > 0 else "" for value in values],
                padding=2,
                fontsize=8,
            )

        axis.set_ylim(0.0, 1.0)
        axis.set_ylabel("Frequency")
        axis.set_title(f"{rank.replace('rank', 'Rank ')} frequency")
        axis.grid(axis="y", alpha=0.25)
        axis.set_axisbelow(True)
        axis.legend(loc="upper right", ncols=3)

    axes[-1].set_xticks(x_positions)
    axes[-1].set_xticklabels(
        [f"{metric}\n{METRIC_NAMES[metric]}" for metric in METRIC_KEYS],
        fontsize=9,
    )
    figure.suptitle(
        f"MITFLD LLM segmentation ranking results (valid samples: {valid_n})",
        fontsize=15,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


@click.command()
@click.option(
    "--responses_jsonl",
    type=str,
    required=True,
    help="Path to responses_*.jsonl generated by mitfld_llm_rank.py",
)
@click.option(
    "--output_dir",
    type=str,
    default="artifacts/mitfld_llm_eval",
    show_default=True,
)
def main(responses_jsonl: str, output_dir: str) -> None:
    responses_path = Path(responses_jsonl)
    if not responses_path.exists():
        raise click.BadParameter(f"responses_jsonl not found: {responses_jsonl}", param_hint="responses_jsonl")

    rows = _read_jsonl(responses_path)
    valid_rows = _extract_valid_rows(rows)
    valid_n = len(valid_rows)

    counts = _count_frequency(valid_rows)

    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    run_stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    long_csv = out_root / f"rank_frequency_long_{run_stamp}.csv"
    wide_csv = out_root / f"rank_frequency_wide_{run_stamp}.csv"
    summary_json = out_root / f"rank_frequency_summary_{run_stamp}.json"
    plot_path = out_root / f"rank_frequency_plot_{run_stamp}.png"

    _write_summary_long(long_csv, counts, valid_n)
    _write_summary_wide(wide_csv, counts, valid_n)
    _write_frequency_plot(plot_path, counts, valid_n)

    invalid_n = len(rows) - valid_n
    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "input": {
            "responses_jsonl": str(responses_path),
            "total_rows": len(rows),
            "valid_rows": valid_n,
            "invalid_rows": invalid_n,
        },
        "metrics": METRIC_KEYS,
        "annotators": ANNOTATORS,
        "outputs": {
            "rank_frequency_long_csv": str(long_csv),
            "rank_frequency_wide_csv": str(wide_csv),
            "rank_frequency_plot": str(plot_path),
        },
    }
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"Saved long summary CSV: {long_csv}")
    print(f"Saved wide summary CSV: {wide_csv}")
    print(f"Saved visualization: {plot_path}")
    print(f"Saved summary JSON: {summary_json}")


if __name__ == "__main__":
    main()
