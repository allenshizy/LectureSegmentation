"""Pre-training entrypoint for the STB backbone (SOP + MSR objectives).

Pre-trains the GlobalTransformer (and optionally the SentenceEncoder) on raw
lecture transcripts from MITFLD — no segment labels required.  After
pre-training the encoder and transformer weights are saved as checkpoints that
can be passed directly to ``LectureSegmentationModel`` via its
``encoder_checkpoint`` / ``transformer_checkpoint`` constructor arguments.

Quick start::

    python pretrain_stb.py \\
        --dataset_path /data/mitfld \\
        --output_dir pretrained/stb_pretrain \\
        --epochs 20

Add extra data sources later by repeating ``--dataset_path`` (all paths are
loaded and merged into a single :class:`LecturePretrainDataset`)::

    python pretrain_stb.py \\
        --dataset_path /data/mitfld \\
        --dataset_path /data/avlecture \\
        --output_dir pretrained/stb_pretrain \\
        --epochs 30
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

import click
import torch
from torch.utils.data import DataLoader, random_split

from kpi.datasets.mitfld import MITFLD
from kpi.models.STB import STBPretrainingModel
from kpi.utils.logconfig import setup_logging
from kpi.utils.stb_pretrain import (
    LecturePretrainDataset,
    pretrain_collate_fn,
    run_pretrain_epoch,
)
from kpi.utils.stb_supervised import parse_json_kwargs, resolve_device

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@click.command()
@click.option(
    "--dataset_path",
    "dataset_paths",
    multiple=True,
    required=False,
    type=str,
    help=(
        "Path to a raw MITFLD dataset root directory. Repeat the flag to merge "
        "multiple sources, e.g. --dataset_path /data/mitfld."
    ),
)
@click.option(
    "--processed_dataset_path",
    "processed_dataset_paths",
    multiple=True,
    required=False,
    type=str,
    help=(
        "Path to a processed .pt dataset file created by export_text_label_dataset.py. "
        "Can be repeated to merge multiple processed files."
    ),
)
@click.option(
    "--output_dir",
    default="pretrained/stb_pretrain",
    show_default=True,
    type=str,
    help="Directory where checkpoints and logs are written.",
)
@click.option(
    "--epochs",
    default=20,
    show_default=True,
    type=int,
    help="Number of pre-training epochs.",
)
@click.option(
    "--batch_size",
    default=8,
    show_default=True,
    type=int,
    help="Number of lectures per mini-batch.",
)
@click.option(
    "--lr",
    default=1e-4,
    show_default=True,
    type=float,
    help="AdamW learning rate.",
)
@click.option(
    "--weight_decay",
    default=1e-2,
    show_default=True,
    type=float,
    help="AdamW weight decay.",
)
@click.option(
    "--val_ratio",
    default=0.1,
    show_default=True,
    type=float,
    help="Fraction of data reserved for validation (0 to disable).",
)
@click.option(
    "--mask_rate",
    default=0.15,
    show_default=True,
    type=float,
    help="Fraction of sentence positions masked for MSR.",
)
@click.option(
    "--sop_prob",
    default=0.5,
    show_default=True,
    type=float,
    help="Per-document probability of applying SOP block-shuffling.",
)
@click.option(
    "--sop_shuffle_ratio",
    default=0.15,
    show_default=True,
    type=float,
    help="Fraction of total document length to shuffle as a contiguous block for SOP.",
)
@click.option(
    "--sop_weight",
    default=1.0,
    show_default=True,
    type=float,
    help="Loss weight for the SOP objective.",
)
@click.option(
    "--msr_weight",
    default=1.0,
    show_default=True,
    type=float,
    help="Loss weight for the MSR objective.",
)
@click.option(
    "--grad_clip_norm",
    default=1.0,
    show_default=True,
    type=float,
    help="Max gradient norm for clipping (0 to disable).",
)
@click.option(
    "--num_workers",
    default=0,
    show_default=True,
    type=int,
    help="DataLoader worker processes.",
)
@click.option(
    "--device",
    default="auto",
    show_default=True,
    type=str,
    help="Torch device: 'auto', 'cuda', 'cpu', etc.",
)
@click.option(
    "--seed",
    default=2024,
    show_default=True,
    type=int,
    help="Random seed.",
)
@click.option(
    "--save_every",
    default=5,
    show_default=True,
    type=int,
    help="Save a checkpoint every N epochs (0 = only save best and final).",
)
@click.option(
    "--encoder_kwargs",
    default="{}",
    show_default=True,
    type=str,
    help="JSON dict forwarded to SentenceEncoder() constructor.",
)
@click.option(
    "--transformer_kwargs",
    default="{}",
    show_default=True,
    type=str,
    help="JSON dict forwarded to GlobalTransformer() constructor.",
)
@click.option(
    "--encoder_checkpoint",
    default=None,
    type=str,
    help="Path to an existing SentenceEncoder checkpoint to resume from.",
)
@click.option(
    "--transformer_checkpoint",
    default=None,
    type=str,
    help="Path to an existing GlobalTransformer checkpoint to resume from.",
)
@click.option(
    "--log_level",
    default="DID",
    show_default=True,
)
def main(
    dataset_paths: tuple[str, ...],
    processed_dataset_paths: tuple[str, ...],
    output_dir: str,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    val_ratio: float,
    mask_rate: float,
    sop_prob: float,
    sop_shuffle_ratio: float,
    sop_weight: float,
    msr_weight: float,
    grad_clip_norm: float,
    num_workers: int,
    device: str,
    seed: int,
    save_every: int,
    encoder_kwargs: str,
    transformer_kwargs: str,
    encoder_checkpoint: str | None,
    transformer_checkpoint: str | None,
    log_level: str,
) -> None:
    torch.manual_seed(seed)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(log_level, output_dir)

    dev = resolve_device(device)
    enc_kw = parse_json_kwargs("encoder_kwargs", encoder_kwargs)
    tfm_kw = parse_json_kwargs("transformer_kwargs", transformer_kwargs)
    clip = grad_clip_norm if grad_clip_norm > 0 else None

    # ------------------------------------------------------------------
    # Build dataset
    # ------------------------------------------------------------------
    if not dataset_paths and not processed_dataset_paths:
        raise click.UsageError(
            "Must specify at least one of --dataset_path or --processed_dataset_path."
        )

    all_lectures: list[list[str]] = []
    if dataset_paths:
        logger.info("Loading %d raw dataset source(s)…", len(dataset_paths))
        all_datasets = [MITFLD(p) for p in dataset_paths]
        all_lectures.extend(LecturePretrainDataset.from_datasets(all_datasets).lectures)

    if processed_dataset_paths:
        logger.info("Loading %d processed dataset source(s)…", len(processed_dataset_paths))
        all_lectures.extend(
            LecturePretrainDataset.from_processed_files(list(processed_dataset_paths)).lectures
        )

    pretrain_ds = LecturePretrainDataset(all_lectures)
    logger.info("Total lectures available for pre-training: %d", len(pretrain_ds))

    if len(pretrain_ds) < 2:
        raise RuntimeError(
            "Pre-training dataset has fewer than 2 documents.  "
            "Check that the dataset path is correct and contains transcripts."
        )

    # Optional validation split
    if val_ratio > 0.0 and len(pretrain_ds) >= 4:
        n_val = max(1, int(len(pretrain_ds) * val_ratio))
        n_train = len(pretrain_ds) - n_val
        train_ds, val_ds = random_split(
            pretrain_ds,
            [n_train, n_val],
            generator=torch.Generator().manual_seed(seed),
        )
        logger.info("Train=%d  Val=%d", n_train, n_val)
    else:
        train_ds = pretrain_ds
        val_ds = None
        logger.info("No validation split (val_ratio=%.2f or dataset too small)", val_ratio)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=pretrain_collate_fn,
    )
    val_loader: DataLoader | None = None
    if val_ds is not None:
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=pretrain_collate_fn,
        )

    # ------------------------------------------------------------------
    # Build model
    # ------------------------------------------------------------------
    from kpi.models.STB import SentenceEncoder, GlobalTransformer

    encoder = (
        SentenceEncoder.load_checkpoint(encoder_checkpoint)
        if encoder_checkpoint
        else SentenceEncoder(**enc_kw)
    )
    transformer = (
        GlobalTransformer.load_checkpoint(transformer_checkpoint)
        if transformer_checkpoint
        else GlobalTransformer(**tfm_kw)
    )

    model = STBPretrainingModel(
        encoder=encoder,
        transformer=transformer,
        mask_rate=mask_rate,
        sop_prob=sop_prob,
        sop_shuffle_ratio=sop_shuffle_ratio,
    ).to(dev)

    # Only optimise parameters that require grad (SBERT is frozen by default)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    logger.info(
        "Trainable parameters: %d tensors / %.2fM elements",
        len(trainable_params),
        sum(p.numel() for p in trainable_params) / 1e6,
    )

    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    best_val_loss = float("inf")
    history: list[dict] = []

    for epoch in range(1, epochs + 1):
        train_metrics = run_pretrain_epoch(
            model,
            train_loader,
            device=dev,
            optimizer=optimizer,
            sop_weight=sop_weight,
            msr_weight=msr_weight,
            grad_clip_norm=clip,
        )

        row: dict = {"epoch": epoch, **{f"train_{k}": v for k, v in train_metrics.items()}}

        if val_loader is not None:
            with torch.no_grad():
                val_metrics = run_pretrain_epoch(
                    model,
                    val_loader,
                    device=dev,
                    optimizer=None,
                    sop_weight=sop_weight,
                    msr_weight=msr_weight,
                    fixed_seed=seed,
                )
            row.update({f"val_{k}": v for k, v in val_metrics.items()})

            val_loss = val_metrics["loss"]
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                _save_backbone(model, out_dir / "best", epoch)
                logger.info("Epoch %d: new best val_loss=%.4f — saved best checkpoint", epoch, val_loss)

        history.append(row)

        # Console summary
        parts = [f"Epoch {epoch}/{epochs}"]
        parts.append(
            f"train loss={train_metrics['loss']:.4f} "
            f"(sop={train_metrics['sop_loss']:.4f} msr={train_metrics['msr_loss']:.4f})"
        )
        if val_loader is not None:
            parts.append(
                f"val loss={val_metrics['loss']:.4f} "  # type: ignore[possibly-undefined]
                f"(sop={val_metrics['sop_loss']:.4f} msr={val_metrics['msr_loss']:.4f})"
            )
        print(" | ".join(parts))

        # Periodic checkpoint
        if save_every > 0 and epoch % save_every == 0:
            _save_backbone(model, out_dir / f"epoch_{epoch:04d}", epoch)

    # Final checkpoint
    _save_backbone(model, out_dir / "final", epochs)

    # Training history
    history_path = out_dir / "pretrain_history.json"
    history_path.write_text(json.dumps(history, indent=2))
    logger.info("Pre-training complete.  History saved to %s", history_path)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _save_backbone(model: STBPretrainingModel, directory: Path, epoch: int) -> None:
    """Save encoder + transformer checkpoints into *directory*."""
    directory.mkdir(parents=True, exist_ok=True)
    enc_path = directory / "encoder.pt"
    tfm_path = directory / "transformer.pt"
    model.encoder.save_checkpoint(enc_path)
    model.transformer.save_checkpoint(tfm_path)
    meta = {"epoch": epoch, "saved_at": datetime.now().isoformat()}
    (directory / "meta.json").write_text(json.dumps(meta, indent=2))
    logger.info("Saved backbone checkpoint (epoch %d) to %s", epoch, directory)


if __name__ == "__main__":
    main()
