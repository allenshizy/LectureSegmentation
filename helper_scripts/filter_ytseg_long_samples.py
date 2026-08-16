from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter YTSeg processed samples whose sentence count exceeds a threshold."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="processed_dataset/ytseg_text_ref",
        help="Directory containing ytseg_text_*.pt files.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="processed_dataset/ytseg_text_ref_max3500",
        help="Directory to write filtered .pt files.",
    )
    parser.add_argument(
        "--max_sentences",
        type=int,
        default=3500,
        help="Drop samples with sentence count greater than this value.",
    )
    return parser.parse_args()


def filter_payload(payload: dict[str, Any], max_sentences: int) -> tuple[dict[str, Any], dict[str, Any]]:
    samples = payload.get("samples", [])
    if not isinstance(samples, list):
        raise ValueError("payload['samples'] must be a list")

    kept_samples: list[dict[str, Any]] = []
    removed_lengths: list[int] = []

    for sample in samples:
        text_list = sample.get("text", [])
        if not isinstance(text_list, list):
            text_list = list(text_list) if text_list is not None else []

        if len(text_list) > max_sentences:
            removed_lengths.append(len(text_list))
            continue
        kept_samples.append(sample)

    filtered_payload = dict(payload)
    filtered_payload["samples"] = kept_samples
    filtered_payload["max_sentences"] = max_sentences
    filtered_payload["filtered"] = {
        "removed_samples": len(removed_lengths),
        "kept_samples": len(kept_samples),
    }

    report = {
        "input_samples": len(samples),
        "kept_samples": len(kept_samples),
        "removed_samples": len(removed_lengths),
        "removed_lengths": removed_lengths,
    }
    return filtered_payload, report


def main() -> None:
    args = parse_args()
    if args.max_sentences <= 0:
        raise ValueError("--max_sentences must be > 0")

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pt_files = sorted(input_dir.glob("*.pt"))
    if not pt_files:
        raise FileNotFoundError(f"No .pt files found in {input_dir}")

    summary: dict[str, Any] = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "max_sentences": args.max_sentences,
        "files": {},
    }

    for pt_file in pt_files:
        payload = torch.load(pt_file, map_location="cpu")
        if not isinstance(payload, dict) or "samples" not in payload:
            raise ValueError(f"Unsupported payload format in {pt_file}")

        filtered_payload, report = filter_payload(payload, max_sentences=args.max_sentences)
        out_path = output_dir / pt_file.name
        torch.save(filtered_payload, out_path)
        summary["files"][pt_file.name] = report
        print(
            f"{pt_file.name}: kept={report['kept_samples']} removed={report['removed_samples']} -> {out_path}"
        )

    report_path = output_dir / "filter_report.json"
    with report_path.open("w", encoding="utf-8") as fp:
        json.dump(summary, fp, ensure_ascii=False, indent=2)
    print(f"Saved report to {report_path}")


if __name__ == "__main__":
    main()