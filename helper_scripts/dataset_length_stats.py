from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import hashlib

import matplotlib.pyplot as plt
import numpy as np
import torch


DEFAULT_DATASET_DIRS = [
    "processed_dataset/mitfld_split",
    "processed_dataset/openphi_textbook_merged_short",
    "processed_dataset/ytseg_text_ref",
]


@dataclass
class DatasetStats:
    name: str
    files: list[str] = field(default_factory=list)
    split_counts: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    sample_lengths: list[int] = field(default_factory=list)
    sentence_lengths: list[int] = field(default_factory=list)
    split_sample_lengths: dict[str, list[int]] = field(default_factory=lambda: defaultdict(list))
    split_sentence_lengths: dict[str, list[int]] = field(default_factory=lambda: defaultdict(list))
    split_boundary_positive: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    split_boundary_total: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    split_labeled_samples: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    boundary_positive_total: int = 0
    boundary_total: int = 0
    labeled_samples: int = 0
    total_samples: int = 0
    total_sentences: int = 0
    unique_sample_signatures: set[str] = field(default_factory=set)
    unique_sentences: set[str] = field(default_factory=set)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize document/sentence length distributions for processed .pt datasets "
            "and save plots."
        )
    )
    parser.add_argument(
        "--dataset",
        dest="datasets",
        action="append",
        help=(
            "Dataset directory or .pt file path. Repeat multiple times. "
            "If omitted, uses the 3 default dataset directories in this repo."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="artifacts/dataset_stats",
        help="Output directory for summary json and png charts.",
    )
    parser.add_argument(
        "--sentence_length_unit",
        choices=["char", "token"],
        default="char",
        help="How to measure each sentence length.",
    )
    return parser.parse_args()


def collect_pt_files(path_str: str) -> list[Path]:
    path = Path(path_str)
    if path.is_file():
        return [path] if path.suffix.lower() == ".pt" else []
    if path.is_dir():
        return sorted(path.glob("*.pt"))
    return []


def load_samples(pt_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    payload = torch.load(pt_path, map_location="cpu")
    if isinstance(payload, dict) and "samples" in payload:
        samples = payload["samples"]
        return samples, payload
    if isinstance(payload, list):
        return payload, None
    raise ValueError(f"Unsupported payload format in: {pt_path}")


def infer_split(pt_path: Path, payload: dict[str, Any] | None) -> str:
    if isinstance(payload, dict):
        split = str(payload.get("split", "")).strip().lower()
        if split in {"train", "test", "validation", "val", "dev"}:
            if split in {"val", "dev"}:
                return "validation"
            return split

    lower_name = pt_path.name.lower()
    if "validation" in lower_name or "_val" in lower_name or "-val" in lower_name or "_dev" in lower_name:
        return "validation"
    if "test" in lower_name:
        return "test"
    if "train" in lower_name:
        return "train"
    return "unknown"


def sentence_length(sentence: Any, unit: str) -> int:
    text = str(sentence).strip()
    if not text:
        return 0
    if unit == "token":
        return len(text.split())
    return len(text)


def to_label_list(value: Any) -> list[int] | None:
    if value is None:
        return None
    if torch.is_tensor(value):
        return [int(x) for x in value.detach().cpu().view(-1).tolist()]
    if isinstance(value, (list, tuple)):
        labels: list[int] = []
        for item in value:
            try:
                labels.append(int(item))
            except Exception:
                return None
        return labels
    return None


def sample_signature(text_list: list[Any]) -> str:
    joined = "\n".join(str(x) for x in text_list)
    return hashlib.sha1(joined.encode("utf-8")).hexdigest()


def summarize(values: list[int]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {
            "count": 0,
            "min": 0,
            "max": 0,
            "mean": 0.0,
            "median": 0.0,
            "p90": 0.0,
            "p95": 0.0,
            "p99": 0.0,
        }
    return {
        "count": int(arr.size),
        "min": int(arr.min()),
        "max": int(arr.max()),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
    }


def build_hist_bins(values: list[int], max_bins: int = 80) -> np.ndarray:
    if not values:
        return np.arange(0, 2)
    max_value = max(values)
    if max_value <= max_bins:
        return np.arange(0, max_value + 2)
    return np.linspace(0, max_value, num=max_bins + 1)


def plot_overlay(stats_list: list[DatasetStats], output_dir: Path, sent_unit: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    all_doc_lengths = [v for st in stats_list for v in st.sample_lengths]
    all_sent_lengths = [v for st in stats_list for v in st.sentence_lengths]
    doc_bins = build_hist_bins(all_doc_lengths)
    sent_bins = build_hist_bins(all_sent_lengths)

    for st in stats_list:
        axes[0].hist(
            st.sample_lengths,
            bins=doc_bins,
            density=True,
            histtype="step",
            linewidth=1.6,
            label=f"{st.name} (n={len(st.sample_lengths)})",
        )
        axes[1].hist(
            st.sentence_lengths,
            bins=sent_bins,
            density=True,
            histtype="step",
            linewidth=1.6,
            label=f"{st.name} (n={len(st.sentence_lengths)})",
        )

    axes[0].set_title("Sample Length Distribution")
    axes[0].set_xlabel("Sentences per sample")
    axes[0].set_ylabel("Density")
    axes[0].grid(alpha=0.25)

    axes[1].set_title(f"Sentence Length Distribution ({sent_unit})")
    axes[1].set_xlabel(f"Sentence length ({sent_unit})")
    axes[1].set_ylabel("Density")
    axes[1].set_yscale("log")
    axes[1].grid(alpha=0.25)

    for ax in axes:
        ax.legend()

    fig.tight_layout()
    fig.savefig(output_dir / "length_distributions_overlay.png", dpi=180)
    plt.close(fig)


def plot_per_dataset(stats_list: list[DatasetStats], output_dir: Path, sent_unit: str) -> None:
    for st in stats_list:
        fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
        axes[0].hist(st.sample_lengths, bins=build_hist_bins(st.sample_lengths), color="#4C72B0", alpha=0.85)
        axes[0].set_title(f"{st.name} - Sample Length")
        axes[0].set_xlabel("Sentences per sample")
        axes[0].set_ylabel("Count")
        axes[0].grid(alpha=0.25)

        axes[1].hist(st.sentence_lengths, bins=build_hist_bins(st.sentence_lengths), color="#DD8452", alpha=0.85)
        axes[1].set_title(f"{st.name} - Sentence Length ({sent_unit})")
        axes[1].set_xlabel(f"Sentence length ({sent_unit})")
        axes[1].set_ylabel("Count")
        axes[1].set_yscale("log")
        axes[1].grid(alpha=0.25)

        fig.tight_layout()
        fig.savefig(output_dir / f"{st.name}_length_distributions.png", dpi=180)
        plt.close(fig)


def plot_split_overview(stats_list: list[DatasetStats], output_dir: Path) -> None:
    for st in stats_list:
        splits = sorted(st.split_counts.keys())
        if not splits:
            continue

        counts = [st.split_counts[s] for s in splits]
        boundary_rates: list[float] = []
        for s in splits:
            total = st.split_boundary_total.get(s, 0)
            pos = st.split_boundary_positive.get(s, 0)
            rate = (pos / total) if total > 0 else 0.0
            boundary_rates.append(rate)

        fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
        axes[0].bar(splits, counts, color="#4C72B0", alpha=0.9)
        axes[0].set_title(f"{st.name} - Samples by Split")
        axes[0].set_xlabel("Split")
        axes[0].set_ylabel("Samples")
        axes[0].grid(axis="y", alpha=0.25)

        axes[1].bar(splits, boundary_rates, color="#55A868", alpha=0.9)
        axes[1].set_title(f"{st.name} - Boundary Positive Rate by Split")
        axes[1].set_xlabel("Split")
        axes[1].set_ylabel("Positive rate")
        axes[1].set_ylim(0, 1)
        axes[1].grid(axis="y", alpha=0.25)

        fig.tight_layout()
        fig.savefig(output_dir / f"{st.name}_split_overview.png", dpi=180)
        plt.close(fig)


def build_split_summary(st: DatasetStats) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for split in sorted(st.split_counts.keys()):
        pos = st.split_boundary_positive.get(split, 0)
        total = st.split_boundary_total.get(split, 0)
        rate = float(pos / total) if total > 0 else None
        result[split] = {
            "num_samples": int(st.split_counts.get(split, 0)),
            "sample_length_summary": summarize(st.split_sample_lengths.get(split, [])),
            "sentence_length_summary": summarize(st.split_sentence_lengths.get(split, [])),
            "boundary": {
                "labeled_samples": int(st.split_labeled_samples.get(split, 0)),
                "positive_labels": int(pos),
                "total_labels": int(total),
                "positive_rate": rate,
            },
        }
    return result


def normalize_dataset_name(path_str: str) -> str:
    p = Path(path_str)
    if p.is_dir():
        return p.name
    return p.stem


def main() -> None:
    args = parse_args()

    dataset_inputs = args.datasets if args.datasets else DEFAULT_DATASET_DIRS
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stats_map: dict[str, DatasetStats] = {}

    for dataset_input in dataset_inputs:
        dataset_name = normalize_dataset_name(dataset_input)
        pt_files = collect_pt_files(dataset_input)
        if not pt_files:
            print(f"[WARN] No .pt files found in: {dataset_input}")
            continue

        stats = DatasetStats(name=dataset_name)
        stats.files = [str(p) for p in pt_files]

        for pt_file in pt_files:
            samples, payload = load_samples(pt_file)
            split_name = infer_split(pt_file, payload)
            stats.split_counts[split_name] += len(samples)

            for sample in samples:
                text_list = sample.get("text", [])
                if not isinstance(text_list, list):
                    text_list = list(text_list) if text_list is not None else []

                stats.sample_lengths.append(len(text_list))
                stats.split_sample_lengths[split_name].append(len(text_list))
                stats.total_samples += 1
                stats.total_sentences += len(text_list)
                stats.unique_sample_signatures.add(sample_signature(text_list))

                for sentence in text_list:
                    stats.unique_sentences.add(str(sentence).strip())
                    stats.sentence_lengths.append(sentence_length(sentence, args.sentence_length_unit))
                    stats.split_sentence_lengths[split_name].append(sentence_length(sentence, args.sentence_length_unit))

                labels = to_label_list(sample.get("label"))
                if labels is not None and len(labels) > 0:
                    pos = sum(1 for x in labels if x > 0)
                    stats.boundary_positive_total += pos
                    stats.boundary_total += len(labels)
                    stats.labeled_samples += 1
                    stats.split_boundary_positive[split_name] += pos
                    stats.split_boundary_total[split_name] += len(labels)
                    stats.split_labeled_samples[split_name] += 1

        stats_map[dataset_name] = stats

    stats_list = list(stats_map.values())
    if not stats_list:
        raise RuntimeError("No valid datasets were loaded. Check --dataset paths.")

    summary = {
        "sentence_length_unit": args.sentence_length_unit,
        "datasets": {},
    }

    for st in stats_list:
        sample_dup_count = st.total_samples - len(st.unique_sample_signatures)
        sentence_dup_count = st.total_sentences - len(st.unique_sentences)
        boundary_rate = (
            float(st.boundary_positive_total / st.boundary_total)
            if st.boundary_total > 0
            else None
        )
        summary["datasets"][st.name] = {
            "files": st.files,
            "split_counts": dict(st.split_counts),
            "sample_length_summary": summarize(st.sample_lengths),
            "sentence_length_summary": summarize(st.sentence_lengths),
            "split_stats": build_split_summary(st),
            "boundary_overall": {
                "labeled_samples": int(st.labeled_samples),
                "positive_labels": int(st.boundary_positive_total),
                "total_labels": int(st.boundary_total),
                "positive_rate": boundary_rate,
            },
            "duplication": {
                "total_samples": int(st.total_samples),
                "unique_samples": int(len(st.unique_sample_signatures)),
                "duplicate_samples": int(sample_dup_count),
                "duplicate_sample_rate": float(sample_dup_count / st.total_samples) if st.total_samples > 0 else 0.0,
                "total_sentences": int(st.total_sentences),
                "unique_sentences": int(len(st.unique_sentences)),
                "duplicate_sentences": int(sentence_dup_count),
                "duplicate_sentence_rate": float(sentence_dup_count / st.total_sentences) if st.total_sentences > 0 else 0.0,
            },
        }

    with (output_dir / "length_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    plot_overlay(stats_list, output_dir=output_dir, sent_unit=args.sentence_length_unit)
    plot_per_dataset(stats_list, output_dir=output_dir, sent_unit=args.sentence_length_unit)
    plot_split_overview(stats_list, output_dir=output_dir)

    print("Saved:")
    print(f"  - {output_dir / 'length_summary.json'}")
    print(f"  - {output_dir / 'length_distributions_overlay.png'}")
    for st in stats_list:
        print(f"  - {output_dir / f'{st.name}_length_distributions.png'}")
        print(f"  - {output_dir / f'{st.name}_split_overview.png'}")

    print("\nQuick summary:")
    for st in stats_list:
        sample_s = summarize(st.sample_lengths)
        sent_s = summarize(st.sentence_lengths)
        boundary_rate = (
            (st.boundary_positive_total / st.boundary_total) if st.boundary_total > 0 else None
        )
        dup_sample_rate = (st.total_samples - len(st.unique_sample_signatures)) / st.total_samples if st.total_samples > 0 else 0.0
        print(
            f"[{st.name}] samples={sample_s['count']} "
            f"sample_len(mean/med/p95)={sample_s['mean']:.2f}/{sample_s['median']:.2f}/{sample_s['p95']:.2f} "
            f"sentence_len(mean/med/p95)={sent_s['mean']:.2f}/{sent_s['median']:.2f}/{sent_s['p95']:.2f} "
            f"boundary_pos_rate={boundary_rate if boundary_rate is not None else 'NA'} "
            f"dup_sample_rate={dup_sample_rate:.4f}"
        )


if __name__ == "__main__":
    main()
