"""Supervised training entrypoint for STB on MITFLD."""

from __future__ import annotations

import copy
import json
import logging
from datetime import datetime
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
import click

from kpi.datasets.mitfld import MITFLD

from kpi.utils import TextLabelSequenceDataset, set_seed, text_label_collate_fn
from kpi.utils.processed_split import partition_processed_paths
from kpi.utils.stb_supervised import (
    ProcessedTextLabelDataset,
    compute_pos_weight,
    evaluate_boundary_metrics,
    evaluate_boundary_metrics_processed,
    load_processed_samples,
    parse_json_kwargs,
    processed_text_label_collate_fn,
    resolve_device,
    run_epoch,
    select_best_boundary_threshold,
    select_best_boundary_threshold_processed,
    validate_split_ratios,
)
from kpi.utils.logconfig import setup_logging

from kpi.models.STB import LectureSegmentationModel, LinearBoundaryDetector

SEED = 2024
logger = logging.getLogger(__name__)


def _build_threshold_grid(min_value: float, max_value: float, step: float) -> list[float]:
    thresholds: list[float] = []
    current = min_value
    while current <= max_value + 1e-12:
        thresholds.append(float(round(current, 10)))
        current += step
    if not thresholds:
        thresholds = [float(min_value)]
    return thresholds


def _save_model_parts(model: LectureSegmentationModel, target_dir: Path) -> dict[str, str]:
    target_dir.mkdir(parents=True, exist_ok=True)
    encoder_path = target_dir / "encoder.pt"
    transformer_path = target_dir / "transformer.pt"
    detector_path = target_dir / "detector.pt"

    model.encoder.save_checkpoint(encoder_path)
    model.transformer.save_checkpoint(transformer_path)
    model.detector.save_checkpoint(detector_path)
    return {
        "encoder": str(encoder_path),
        "transformer": str(transformer_path),
        "detector": str(detector_path),
    }

@click.command()
@click.option(
    "--dataset_path",
    required=False,
    type=str,
    help="Path to the dataset root directory.",
)
@click.option(
    "--processed_dataset_path",
    "processed_dataset_paths",
    type=str,
    multiple=True,
    help=(
        "Path to a processed text-label dataset .pt file. Repeat to provide "
        "pre-split train/validation/test files."
    ),
)
@click.option(
    "--encoder_weights_path",
    type=str,
)
@click.option(
    "--transformer_weights_path",
    type=str,
)
@click.option(
    "--detector_weights_path",
    type=str,
)
@click.option(
    "--encoder_kwargs",
    type=str,
    default="{}",
    help="JSON string of keyword arguments for the SentenceEncoder.",
)
@click.option(
    "--transformer_kwargs",
    type=str,
    default="{}",
    help="JSON string of keyword arguments for the GlobalTransformer.",
)
@click.option(
    "--detector_kwargs",
    type=str,
    default="{}",
    help="JSON string of keyword arguments for the BoundaryDetector.",
)
@click.option(
    "--detector_mode",
    type=click.Choice(["bilstm", "mlp"], case_sensitive=False),
    default="bilstm",
    show_default=True,
    help="Select whether the detector uses the existing BiLSTM path or a pure MLP path.",
)
@click.option(
    "--log_level",
    type=str,
    default="DID",
    help="Logging level (e.g., INFO, DEBUG, WARNING, ERROR, CRITICAL).",
)
@click.option("--seed", type=int, default=SEED, show_default=True)
@click.option("--split_seed", type=int, default=SEED, show_default=True)
@click.option("--train_ratio", type=float, default=0.7, show_default=True)
@click.option("--val_ratio", type=float, default=0.15, show_default=True)
@click.option("--test_ratio", type=float, default=0.15, show_default=True)
@click.option("--epochs", type=int, default=20, show_default=True)
@click.option("--batch_size", type=int, default=4, show_default=True)
@click.option("--lr", type=float, default=1e-4, show_default=True)
@click.option(
    "--transformer_lr",
    type=float,
    default=None,
    help="Independent learning rate for GlobalTransformer. Defaults to --lr when omitted.",
)
@click.option("--weight_decay", type=float, default=1e-2, show_default=True)
@click.option("--grad_clip_norm", type=float, default=1.0, show_default=True)
@click.option("--boundary_threshold", type=float, default=0.5, show_default=True)
@click.option("--local_max_k", type=int, default=2, show_default=True)
@click.option("--threshold_min", type=float, default=0.0, show_default=True)
@click.option("--threshold_max", type=float, default=1.0, show_default=True)
@click.option("--threshold_step", type=float, default=0.05, show_default=True)
@click.option("--f1_threshold", type=float, default=10, show_default=True)
@click.option("--iou_fps", type=int, default=10, show_default=True)
@click.option("--mof_fps", type=int, default=10, show_default=True)
@click.option("--device", type=str, default="auto", show_default=True)
@click.option("--num_workers", type=int, default=0, show_default=True)
@click.option("--output_dir", type=str, default="artifacts/stb_supervised", show_default=True)
@click.option("--run_name", type=str, default=None)
@click.option("--save_best/--no-save_best", default=True, show_default=True)
@click.option("--save_last/--no-save_last", default=True, show_default=True)
@click.option("--pos_weight_normalizer", type=float, default=1.0, show_default=True)
@click.option("--linear_probe/--no-linear_probe", default=False, show_default=True)
@click.option("--freeze_transformer_epochs", type=int, default=None)
@click.option("--freeze_detector_epochs", type=int, default=None)
def run_supervised_training(
    dataset_path: str | None = None,
    processed_dataset_paths: tuple[str, ...] = (),
    encoder_weights_path: str = None,
    transformer_weights_path: str = None,
    detector_weights_path: str = None,
    encoder_kwargs: str = "{}",
    transformer_kwargs: str = "{}",
    detector_kwargs: str = "{}",
    detector_mode: str = "bilstm",
    log_level: str = "DID",
    seed: int = SEED,
    split_seed: int = SEED,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    epochs: int = 20,
    batch_size: int = 4,
    lr: float = 1e-4,
    transformer_lr: float | None = None,
    pos_weight_normalizer: float = 1.0,
    weight_decay: float = 1e-2,
    grad_clip_norm: float = 1.0,
    boundary_threshold: float = 0.5,
    local_max_k: int = 3,
    threshold_min: float = 0.0,
    threshold_max: float = 1.0,
    threshold_step: float = 0.05,
    f1_threshold: float = 10,
    iou_fps: int = 10,
    mof_fps: int = 10,
    device: str = "auto",
    num_workers: int = 0,
    output_dir: str = "artifacts/stb_supervised",
    run_name: str | None = None,
    save_best: bool = True,
    save_last: bool = True,
    linear_probe: bool = False,
    freeze_transformer_epochs: int | None = None,
    freeze_detector_epochs: int | None = None,
) -> None:
    using_processed_dataset = len(processed_dataset_paths) > 0
    if not using_processed_dataset:
        validate_split_ratios(train_ratio, val_ratio, test_ratio)

    if bool(dataset_path) == using_processed_dataset:
        raise click.BadParameter("Provide exactly one of --dataset_path or --processed_dataset_path")

    if epochs <= 0:
        raise click.BadParameter("epochs must be > 0")
    if batch_size <= 0:
        raise click.BadParameter("batch_size must be > 0")
    if local_max_k < 0:
        raise click.BadParameter("local_max_k must be >= 0")
    if not (0.0 <= threshold_min <= threshold_max <= 1.0):
        raise click.BadParameter("Require 0 <= threshold_min <= threshold_max <= 1")
    if threshold_step <= 0:
        raise click.BadParameter("threshold_step must be > 0")
    if transformer_lr is not None and transformer_lr <= 0:
        raise click.BadParameter("transformer_lr must be > 0 when provided")

    if linear_probe:
        if freeze_transformer_epochs is not None and freeze_transformer_epochs < epochs:
            raise click.BadParameter(
                "linear_probe requires transformer to remain frozen for all epochs; "
                "set --freeze_transformer_epochs >= --epochs or omit it"
            )
        if freeze_detector_epochs is not None and freeze_detector_epochs > 0:
            raise click.BadParameter("linear_probe requires detector to be trainable; set freeze_detector_epochs=0")

    effective_freeze_transformer_epochs = epochs if linear_probe else freeze_transformer_epochs
    effective_freeze_detector_epochs = freeze_detector_epochs

    freeze_plan = {
        "transformer": effective_freeze_transformer_epochs,
        "detector": effective_freeze_detector_epochs,
    }
    for module_name, freeze_epochs in freeze_plan.items():
        if freeze_epochs is not None and freeze_epochs < 0:
            raise click.BadParameter(f"freeze_{module_name}_epochs must be >= 0")

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_stem = run_name or f"stb_mitfld_{timestamp}"

    # Keep all run artifacts under one directory: output_dir/run_name
    output_root = Path(output_dir) / run_stem
    output_root.mkdir(parents=True, exist_ok=True)
    log_path = setup_logging(log_level, str(output_root))

    set_seed(seed)
    model_device = resolve_device(device)
    logger.info("Using device: %s", model_device)
    logger.info("Run output directory: %s", output_root)

    encoder_kwargs_dict = parse_json_kwargs("encoder_kwargs", encoder_kwargs)
    transformer_kwargs_dict = parse_json_kwargs("transformer_kwargs", transformer_kwargs)
    detector_kwargs_dict = parse_json_kwargs("detector_kwargs", detector_kwargs)
    detector_kwargs_dict.setdefault("use_mlp_only", detector_mode.lower() == "mlp")

    test_data = None

    if using_processed_dataset:
        train_paths, val_paths, test_paths = partition_processed_paths(list(processed_dataset_paths))
        logger.info(
            "Resolved processed splits from file names/metadata: train=%d val=%d test=%d",
            len(train_paths),
            len(val_paths),
            len(test_paths),
        )
        if not train_paths or not val_paths or not test_paths:
            raise click.BadParameter(
                "Processed mode requires pre-split train/validation/test files. "
                "Pass files with split metadata or recognizable names (train/val/test)."
            )

        train_samples: list[dict] = []
        val_samples: list[dict] = []
        test_samples: list[dict] = []
        for path in train_paths:
            train_samples.extend(load_processed_samples(path))
        for path in val_paths:
            val_samples.extend(load_processed_samples(path))
        for path in test_paths:
            test_samples.extend(load_processed_samples(path))

        train_ds = ProcessedTextLabelDataset(train_samples)
        val_ds = ProcessedTextLabelDataset(val_samples)
        test_ds = ProcessedTextLabelDataset(test_samples)
        collate_fn = processed_text_label_collate_fn
    else:
        logger.info("Loading MITFLD dataset from %s", dataset_path)
        dataset = MITFLD(dataset_path)
        train_data, val_data, test_data = dataset.random_split(
            [train_ratio, val_ratio, test_ratio],
            seed=split_seed,
        )

        train_ds = TextLabelSequenceDataset(train_data)
        val_ds = TextLabelSequenceDataset(val_data)
        test_ds = TextLabelSequenceDataset(test_data)
        collate_fn = text_label_collate_fn

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
    )

    # model setup
    if linear_probe:
        if detector_weights_path is None:
            detector_module = LinearBoundaryDetector(**detector_kwargs_dict)
        else:
            detector_module = LinearBoundaryDetector.load_checkpoint(detector_weights_path)
        model = LectureSegmentationModel(
            encoder_checkpoint=encoder_weights_path,
            transformer_checkpoint=transformer_weights_path,
            detector=detector_module,
            encoder_kwargs=encoder_kwargs_dict,
            transformer_kwargs=transformer_kwargs_dict,
        ).to(model_device)
    else:
        model = LectureSegmentationModel(
            encoder_checkpoint=encoder_weights_path,
            transformer_checkpoint=transformer_weights_path,
            detector_checkpoint=detector_weights_path,
            encoder_kwargs=encoder_kwargs_dict,
            transformer_kwargs=transformer_kwargs_dict,
            detector_kwargs=detector_kwargs_dict,
        ).to(model_device)
    # SentenceEncoder wraps SBERT inference and is fixed during supervised STB training.
    model.encoder.requires_grad_(False)
    if linear_probe:
        model.transformer.requires_grad_(False)
    logger.info("Model config: %s", model.get_config())
    logger.info("SentenceEncoder is fixed (requires_grad=False) for supervised training")
    if linear_probe:
        logger.info("Linear probe mode enabled: transformer frozen, detector=%s", model.detector.__class__.__name__)

    freeze_log_entries: list[str] = []
    for module_name, freeze_epochs in freeze_plan.items():
        if freeze_epochs is not None and freeze_epochs > 0:
            getattr(model, module_name).requires_grad_(False)
            if freeze_epochs >= epochs:
                freeze_log_entries.append(f"{module_name}:all {epochs} epochs")
            else:
                freeze_log_entries.append(f"{module_name}:first {freeze_epochs} epochs")
    if freeze_log_entries:
        logger.info("Freeze schedule -> %s", ", ".join(freeze_log_entries))


    # training setup
    pos_weight_value = compute_pos_weight(train_ds)
    pos_weight_value = pos_weight_value/pos_weight_normalizer
    pos_weight = torch.tensor(pos_weight_value, dtype=torch.float32, device=model_device)
    logger.info("Using BCE pos_weight=%.4f", pos_weight_value)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")
    effective_transformer_lr = lr if transformer_lr is None else transformer_lr
    if linear_probe:
        param_groups = [
            {"name": "detector", "params": list(model.detector.parameters()), "lr": lr},
        ]
    else:
        param_groups = [
            {
                "name": "transformer",
                "params": list(model.transformer.parameters()),
                "lr": effective_transformer_lr,
            },
            {"name": "detector", "params": list(model.detector.parameters()), "lr": lr},
        ]
    optimizer = torch.optim.AdamW(
        [{"params": group["params"], "lr": group["lr"]} for group in param_groups],
        weight_decay=weight_decay,
    )
    for group in param_groups:
        total_count = sum(param.numel() for param in group["params"])
        trainable_count = sum(param.numel() for param in group["params"] if param.requires_grad)
        logger.info(
            "Optimizer group %s: lr=%.6g trainable=%d total=%d",
            group["name"],
            group["lr"],
            trainable_count,
            total_count,
        )

    best_state = None
    best_val_loss = float("inf")
    best_val_f1 = float("-inf")
    best_epoch = 0
    history: list[dict[str, float | int]] = []
    best_checkpoint_paths: dict[str, str] | None = None
    last_checkpoint_paths: dict[str, str] | None = None
    threshold_candidates = _build_threshold_grid(
        min_value=threshold_min,
        max_value=threshold_max,
        step=threshold_step,
    )

    # training loop
    for epoch_idx in range(1, epochs + 1):
        for module_name, freeze_epochs in freeze_plan.items():
            if (
                freeze_epochs is not None
                and freeze_epochs > 0
                and freeze_epochs < epochs
                and epoch_idx == freeze_epochs + 1
            ):
                getattr(model, module_name).requires_grad_(True)
                logger.info("Unfroze %s at epoch %d", module_name, epoch_idx)

        train_stats = run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            device=model_device,
            optimizer=optimizer,
            grad_clip_norm=grad_clip_norm,
        )
        with torch.no_grad():
            val_stats = run_epoch(
                model=model,
                loader=val_loader,
                criterion=criterion,
                device=model_device,
                optimizer=None,
            )

        if using_processed_dataset:
            (
                epoch_boundary_threshold,
                epoch_val_f1,
                _,
            ) = select_best_boundary_threshold_processed(
                model=model,
                loader=val_loader,
                device=model_device,
                threshold_candidates=threshold_candidates,
                local_max_k=local_max_k,
                f1_threshold=f1_threshold,
            )
        else:
            (
                epoch_boundary_threshold,
                epoch_val_f1,
                _,
            ) = select_best_boundary_threshold(
                model=model,
                dataset_split=val_data,
                loader=val_loader,
                device=model_device,
                threshold_candidates=threshold_candidates,
                local_max_k=local_max_k,
                f1_threshold=f1_threshold,
            )

        history.append(
            {
                "epoch": epoch_idx,
                "train_loss": train_stats["loss"],
                "val_loss": val_stats["loss"],
                "val_f1": epoch_val_f1,
                "val_boundary_threshold": epoch_boundary_threshold,
            }
        )

        logger.info(
            "epoch=%d train_loss=%.6f val_loss=%.6f val_f1=%.6f val_boundary_threshold=%.4f",
            epoch_idx,
            train_stats["loss"],
            val_stats["loss"],
            epoch_val_f1,
            epoch_boundary_threshold,
        )

        if epoch_val_f1 > best_val_f1 or (
            epoch_val_f1 == best_val_f1 and val_stats["loss"] < best_val_loss
        ):
            best_val_loss = val_stats["loss"]
            best_val_f1 = epoch_val_f1
            best_epoch = epoch_idx
            best_state = copy.deepcopy(model.state_dict())
            if save_best:
                best_dir = output_root / "best"
                best_checkpoint_paths = _save_model_parts(model, best_dir)
                logger.info("Saved best checkpoints by val_f1 to %s", best_dir)

    if best_state is not None:
        model.load_state_dict(best_state)

    if save_last:
        last_dir = output_root / "last"
        last_checkpoint_paths = _save_model_parts(model, last_dir)
        logger.info("Saved last checkpoints to %s", last_dir)

    logger.info(
        "Selecting boundary threshold on val split: min=%.3f max=%.3f step=%.3f (%d candidates)",
        threshold_min,
        threshold_max,
        threshold_step,
        len(threshold_candidates),
    )

    selected_boundary_threshold = boundary_threshold
    selected_val_f1 = float("-inf")
    threshold_sweep: list[dict[str, float]] = []
    if using_processed_dataset:
        (
            selected_boundary_threshold,
            selected_val_f1,
            threshold_sweep,
        ) = select_best_boundary_threshold_processed(
            model=model,
            loader=val_loader,
            device=model_device,
            threshold_candidates=threshold_candidates,
            local_max_k=local_max_k,
            f1_threshold=f1_threshold,
        )
    else:
        (
            selected_boundary_threshold,
            selected_val_f1,
            threshold_sweep,
        ) = select_best_boundary_threshold(
            model=model,
            dataset_split=val_data,
            loader=val_loader,
            device=model_device,
            threshold_candidates=threshold_candidates,
            local_max_k=local_max_k,
            f1_threshold=f1_threshold,
        )

    best_val_f1 = selected_val_f1
    logger.info(
        "Selected boundary threshold=%.4f from val (best F1=%.6f)",
        selected_boundary_threshold,
        selected_val_f1,
    )

    if using_processed_dataset:
        test_boundary_scores = evaluate_boundary_metrics_processed(
            model=model,
            loader=test_loader,
            device=model_device,
            boundary_threshold=selected_boundary_threshold,
            local_max_k=local_max_k,
            f1_threshold=f1_threshold,
            iou_fps=iou_fps,
            mof_fps=mof_fps,
        )
    else:
        test_boundary_scores = evaluate_boundary_metrics(
            model=model,
            dataset_split=test_data,
            loader=test_loader,
            device=model_device,
            boundary_threshold=selected_boundary_threshold,
            local_max_k=local_max_k,
            f1_threshold=f1_threshold,
            iou_fps=iou_fps,
            mof_fps=mof_fps,
        )

    logger.info("Test boundary metrics (F1 as main):")
    for metric_name, value in test_boundary_scores.items():
        logger.info("%s = %.6f", metric_name, value)

    summary = {
        "run_name": run_stem,
        "seed": seed,
        "split_seed": split_seed,
        "device": str(model_device),
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "best_val_f1": best_val_f1,
        "selected_boundary_threshold": selected_boundary_threshold,
        "threshold_sweep": threshold_sweep,
        "test_metrics": test_boundary_scores,
        "best_checkpoints": best_checkpoint_paths,
        "last_checkpoints": last_checkpoint_paths,
        "history": history,
        "options": {
            "output_dir": str(output_root),
            "log_path": str(log_path),
            "dataset_path": dataset_path,
            "processed_dataset_paths": list(processed_dataset_paths),
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "transformer_lr": transformer_lr,
            "transformer_lr_effective": effective_transformer_lr,
            "linear_probe": linear_probe,
            "weight_decay": weight_decay,
            "boundary_threshold_init": boundary_threshold,
            "threshold_min": threshold_min,
            "threshold_max": threshold_max,
            "threshold_step": threshold_step,
            "boundary_threshold": selected_boundary_threshold,
            "local_max_k": local_max_k,
            "f1_threshold": f1_threshold,
            "iou_fps": iou_fps,
            "mof_fps": mof_fps,
            "train_ratio": train_ratio,
            "val_ratio": val_ratio,
            "test_ratio": test_ratio,
            "encoder_weights_path": encoder_weights_path,
            "transformer_weights_path": transformer_weights_path,
            "detector_weights_path": detector_weights_path,
            "encoder_trainable": False,
            "freeze_transformer_epochs": effective_freeze_transformer_epochs,
            "freeze_detector_epochs": effective_freeze_detector_epochs,
        },
    }
    summary_path = output_root / "summary.json"
    with summary_path.open("w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2, ensure_ascii=True)
    logger.info("Saved training summary to %s", summary_path)


if __name__ == "__main__":
    run_supervised_training()