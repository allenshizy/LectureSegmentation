from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from kpi.models.STB import GlobalTransformer, STBPretrainingModel, SentenceEncoder
from kpi.utils.processed_split import partition_processed_paths
from kpi.utils.stb_pretrain import LecturePretrainDataset, pretrain_collate_fn
from kpi.utils.stb_supervised import parse_json_kwargs, resolve_device


@dataclass
class TrialResult:
    batch_size: int
    success: bool
    steps: int
    avg_step_time_sec: float
    steps_per_sec: float
    sentences_per_sec: float
    peak_memory_gb: float | None
    oom_error: str | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark candidate batch sizes for STB pretraining on current hardware "
            "and recommend a practical value."
        )
    )
    parser.add_argument(
        "--processed_dataset_path",
        dest="processed_dataset_paths",
        action="append",
        required=True,
        help="Path to a processed .pt dataset file. Repeat to provide multiple files.",
    )
    parser.add_argument(
        "--candidate_batch_sizes",
        type=str,
        default="2,4,6,8,10,12",
        help="Comma-separated batch sizes to test in ascending order.",
    )
    parser.add_argument("--steps", type=int, default=15, help="Measured optimizer steps per trial.")
    parser.add_argument("--warmup_steps", type=int, default=3, help="Warmup steps before timing.")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--mask_rate", type=float, default=0.15)
    parser.add_argument("--sop_prob", type=float, default=0.5)
    parser.add_argument("--sop_shuffle_ratio", type=float, default=0.15)
    parser.add_argument(
        "--encoder_kwargs",
        type=str,
        default="{}",
        help="JSON dict forwarded to SentenceEncoder() constructor.",
    )
    parser.add_argument(
        "--transformer_kwargs",
        type=str,
        default="{}",
        help="JSON dict forwarded to GlobalTransformer() constructor.",
    )
    parser.add_argument("--encoder_checkpoint", type=str, default=None)
    parser.add_argument("--transformer_checkpoint", type=str, default=None)
    parser.add_argument(
        "--use_amp",
        action="store_true",
        help="Enable mixed precision autocast during benchmark steps.",
    )
    parser.add_argument(
        "--target_memory_util",
        type=float,
        default=0.9,
        help="Preferred max GPU memory utilization for recommended batch size.",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default="artifacts/batch_size_benchmark/pretrain_batch_size_report.json",
        help="Path to save the benchmark report as JSON.",
    )
    return parser.parse_args()


def parse_batch_sizes(spec: str) -> list[int]:
    values = [part.strip() for part in spec.split(",") if part.strip()]
    sizes = sorted({int(v) for v in values})
    if not sizes or any(v <= 0 for v in sizes):
        raise ValueError("--candidate_batch_sizes must contain positive integers")
    return sizes


def _compute_losses(out: dict[str, torch.Tensor], lengths: torch.Tensor, device: torch.device) -> torch.Tensor:
    valid: torch.Tensor = out["valid_mask"].to(device)
    sop_labels_valid = out["sop_labels"].to(device)[valid]
    n_pos = sop_labels_valid.sum().clamp(min=1.0)
    n_neg = (valid.sum().float() - n_pos).clamp(min=1.0)
    pos_weight = n_neg / n_pos
    sop_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    if valid.any():
        sop_loss = sop_criterion(out["sop_logits"][valid], sop_labels_valid)
    else:
        sop_loss = torch.tensor(0.0, device=device)

    if out["msr_preds"].size(0) > 0:
        msr_loss = nn.MSELoss()(out["msr_preds"], out["msr_targets"].to(device))
    else:
        msr_loss = torch.tensor(0.0, device=device)
    return sop_loss + msr_loss


def _is_oom_error(exc: BaseException) -> bool:
    text = str(exc).lower()
    return "out of memory" in text or "cuda error: out of memory" in text


def run_trial(
    model: STBPretrainingModel,
    dataset: LecturePretrainDataset,
    *,
    batch_size: int,
    device: torch.device,
    num_workers: int,
    steps: int,
    warmup_steps: int,
    lr: float,
    weight_decay: float,
    use_amp: bool,
) -> TrialResult:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=pretrain_collate_fn,
        drop_last=False,
    )
    iterator = iter(loader)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=weight_decay,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and device.type == "cuda"))

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    measured_steps = 0
    measured_sentences = 0
    step_times: list[float] = []

    total_steps = warmup_steps + steps
    try:
        model.train(True)
        for step_idx in range(total_steps):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                batch = next(iterator)

            lengths = batch["lengths"].to(device)
            start = time.perf_counter()

            with torch.autocast(
                device_type="cuda" if device.type == "cuda" else "cpu",
                dtype=torch.float16 if device.type == "cuda" else torch.bfloat16,
                enabled=(use_amp and device.type == "cuda"),
            ):
                out: dict[str, torch.Tensor] = model(batch["text"], lengths)
                loss = _compute_losses(out, lengths=lengths, device=device)

            optimizer.zero_grad(set_to_none=True)
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - start

            if step_idx >= warmup_steps:
                step_times.append(elapsed)
                measured_steps += 1
                measured_sentences += int(lengths.sum().item())

        avg_step_time = float(sum(step_times) / max(len(step_times), 1))
        steps_per_sec = float(1.0 / avg_step_time) if avg_step_time > 0 else 0.0
        sentences_per_sec = float(measured_sentences / max(sum(step_times), 1e-9))

        peak_gb = None
        if device.type == "cuda":
            peak_gb = float(torch.cuda.max_memory_allocated(device) / (1024**3))

        return TrialResult(
            batch_size=batch_size,
            success=True,
            steps=measured_steps,
            avg_step_time_sec=avg_step_time,
            steps_per_sec=steps_per_sec,
            sentences_per_sec=sentences_per_sec,
            peak_memory_gb=peak_gb,
            oom_error=None,
        )
    except RuntimeError as exc:
        if _is_oom_error(exc):
            if device.type == "cuda":
                torch.cuda.empty_cache()
            return TrialResult(
                batch_size=batch_size,
                success=False,
                steps=0,
                avg_step_time_sec=math.inf,
                steps_per_sec=0.0,
                sentences_per_sec=0.0,
                peak_memory_gb=None,
                oom_error=str(exc),
            )
        raise


def choose_recommendation(
    results: list[TrialResult],
    *,
    total_gpu_gb: float | None,
    target_memory_util: float,
) -> dict[str, Any]:
    successful = [r for r in results if r.success]
    if not successful:
        return {
            "recommended_batch_size": None,
            "reason": "No candidate batch size succeeded.",
            "max_safe_batch_size": None,
            "best_throughput_batch_size": None,
        }

    max_safe = max(successful, key=lambda r: r.batch_size)
    best_tp = max(successful, key=lambda r: r.sentences_per_sec)

    recommended = best_tp
    reason = "Highest measured sentence throughput among successful candidates."

    if total_gpu_gb is not None:
        constrained = [
            r
            for r in successful
            if r.peak_memory_gb is not None and (r.peak_memory_gb / total_gpu_gb) <= target_memory_util
        ]
        if constrained:
            recommended = max(constrained, key=lambda r: r.sentences_per_sec)
            reason = (
                "Highest sentence throughput under target memory utilization "
                f"({target_memory_util:.2f})."
            )

    return {
        "recommended_batch_size": recommended.batch_size,
        "reason": reason,
        "max_safe_batch_size": max_safe.batch_size,
        "best_throughput_batch_size": best_tp.batch_size,
    }


def main() -> None:
    args = parse_args()
    candidate_batch_sizes = parse_batch_sizes(args.candidate_batch_sizes)

    dev = resolve_device(args.device)
    enc_kw = parse_json_kwargs("encoder_kwargs", args.encoder_kwargs)
    tfm_kw = parse_json_kwargs("transformer_kwargs", args.transformer_kwargs)

    train_paths, val_paths, test_paths = partition_processed_paths(list(args.processed_dataset_paths))
    selected_paths = train_paths if train_paths else list(args.processed_dataset_paths)
    if not selected_paths:
        raise ValueError("No usable processed_dataset_path values were provided.")

    print("Using files for benchmark:")
    for path in selected_paths:
        print(f"  - {path}")
    if val_paths or test_paths:
        print(
            f"Ignored split files for speed: val={len(val_paths)} test={len(test_paths)}; "
            "benchmark uses train split only."
        )

    dataset = LecturePretrainDataset.from_processed_files(selected_paths)
    if len(dataset) < 2:
        raise RuntimeError("Need at least 2 lecture samples to benchmark batch size.")

    encoder = (
        SentenceEncoder.load_checkpoint(args.encoder_checkpoint)
        if args.encoder_checkpoint
        else SentenceEncoder(**enc_kw)
    )
    transformer = (
        GlobalTransformer.load_checkpoint(args.transformer_checkpoint)
        if args.transformer_checkpoint
        else GlobalTransformer(**tfm_kw)
    )
    model = STBPretrainingModel(
        encoder=encoder,
        transformer=transformer,
        mask_rate=args.mask_rate,
        sop_prob=args.sop_prob,
        sop_shuffle_ratio=args.sop_shuffle_ratio,
    ).to(dev)

    total_gpu_gb: float | None = None
    if dev.type == "cuda":
        total_gpu_gb = float(torch.cuda.get_device_properties(dev).total_memory / (1024**3))

    print(
        f"\nBenchmark device={dev}, dataset_size={len(dataset)}, "
        f"candidates={candidate_batch_sizes}, warmup={args.warmup_steps}, steps={args.steps}"
    )

    results: list[TrialResult] = []
    for bs in candidate_batch_sizes:
        print(f"\n[Trial] batch_size={bs}")
        result = run_trial(
            model=model,
            dataset=dataset,
            batch_size=bs,
            device=dev,
            num_workers=args.num_workers,
            steps=args.steps,
            warmup_steps=args.warmup_steps,
            lr=args.lr,
            weight_decay=args.weight_decay,
            use_amp=args.use_amp,
        )
        results.append(result)

        if result.success:
            peak_str = f", peak_mem={result.peak_memory_gb:.2f} GB" if result.peak_memory_gb is not None else ""
            print(
                f"  success: steps/s={result.steps_per_sec:.3f}, "
                f"sentences/s={result.sentences_per_sec:.1f}{peak_str}"
            )
        else:
            print("  failed: OOM")

    recommendation = choose_recommendation(
        results,
        total_gpu_gb=total_gpu_gb,
        target_memory_util=args.target_memory_util,
    )

    report = {
        "device": str(dev),
        "total_gpu_gb": total_gpu_gb,
        "dataset_size": len(dataset),
        "candidate_batch_sizes": candidate_batch_sizes,
        "steps": args.steps,
        "warmup_steps": args.warmup_steps,
        "use_amp": bool(args.use_amp),
        "train_files": selected_paths,
        "results": [asdict(r) for r in results],
        "recommendation": recommendation,
    }

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\nRecommendation:")
    print(json.dumps(recommendation, indent=2, ensure_ascii=False))
    print(f"\nSaved report to: {out_path}")


if __name__ == "__main__":
    main()
