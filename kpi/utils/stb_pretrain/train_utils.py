"""Training-loop helpers for STB pre-training (SOP + MSR objectives)."""

from __future__ import annotations

import logging
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


def run_pretrain_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    sop_weight: float = 1.0,
    msr_weight: float = 1.0,
    grad_clip_norm: float | None = None,
    fixed_seed: int | None = None,
    use_amp: bool = False,
    scaler: Any | None = None,
) -> dict[str, float]:
    """Run one pre-training epoch and return average losses.

    Args:
        model: :class:`~kpi.models.STB.STBPretrainingModel` instance.
        loader: DataLoader yielding batches from
            :class:`~kpi.utils.stb_pretrain.LecturePretrainDataset`.
        device: Torch device.
        optimizer: If provided the function runs in *train* mode and updates
            weights; otherwise runs in *eval* mode (no weight updates).
        sop_weight: Scalar multiplier for the SOP loss term.
        msr_weight: Scalar multiplier for the MSR loss term.
        grad_clip_norm: If set, gradient norms are clipped to this value.
        fixed_seed: If provided and optimizer is None (eval mode),
            torch.manual_seed() is set to this value at epoch start,
            ensuring deterministic perturbations for reproducible validation.

    Returns:
        Dict with ``loss``, ``sop_loss``, ``msr_loss`` (all averaged over
        batches).
    """
    is_train = optimizer is not None
    model.train(is_train)
    mode = "train" if is_train else "eval"

    # Set deterministic seed for validation to ensure reproducible perturbations
    if not is_train and fixed_seed is not None:
        torch.manual_seed(fixed_seed)

    logger.debug("Starting pretrain %s epoch over %d batches", mode, len(loader))

    msr_criterion = nn.MSELoss()
    autocast_device = "cuda" if device.type == "cuda" else "cpu"
    autocast_dtype = torch.float16 if device.type == "cuda" else torch.bfloat16
    autocast_ctx = torch.autocast(
        device_type=autocast_device,
        dtype=autocast_dtype,
        enabled=use_amp,
    )

    total_loss = 0.0
    total_sop = 0.0
    total_msr = 0.0
    n_batches = 0

    for batch_idx, batch in enumerate(loader, start=1):
        lengths: torch.Tensor = batch["lengths"].to(device)

        with autocast_ctx:
            # For validation: explicitly enable perturbations with fixed seed for reproducibility
            if not is_train:
                out: dict[str, torch.Tensor] = model(batch["text"], lengths, apply_perturbations=True)
            else:
                out: dict[str, torch.Tensor] = model(batch["text"], lengths)

            # ---- SOP loss (only on valid, non-padded positions) ---------------
            # SOP labels are heavily imbalanced (~7.5% positive when sop_prob=0.5,
            # sop_shuffle_ratio=0.15), so compute pos_weight dynamically per batch
            # to prevent the model from collapsing to predicting all-zero.
            valid: torch.Tensor = out["valid_mask"].to(device)   # [B, L] bool
            sop_labels_valid = out["sop_labels"].to(device)[valid]
            n_pos = sop_labels_valid.sum().clamp(min=1.0)
            n_neg = (valid.sum().float() - n_pos).clamp(min=1.0)
            pos_weight = n_neg / n_pos  # e.g. ~12 when sop_prob=0.5, sop_shuffle_ratio=0.15
            sop_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            if valid.any():
                sop_loss = sop_criterion(out["sop_logits"][valid], sop_labels_valid)
            else:
                sop_loss = torch.tensor(0.0, device=device)

            # ---- MSR loss (only when masked positions exist) ------------------
            if out["msr_preds"].size(0) > 0:
                msr_loss = msr_criterion(
                    out["msr_preds"], out["msr_targets"].to(device)
                )
            else:
                msr_loss = torch.tensor(0.0, device=device)

            loss = sop_weight * sop_loss + msr_weight * msr_loss

        if is_train:
            optimizer.zero_grad()
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            if grad_clip_norm is not None:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

        loss_value = loss.detach().float()
        sop_value = sop_loss.detach().float()
        msr_value = msr_loss.detach().float()

        total_loss += loss_value.item()
        total_sop += sop_value.item()
        total_msr += msr_value.item()
        n_batches += 1

        if batch_idx % 50 == 0:
            logger.debug(
                "Batch %d/%d | loss=%.4f sop=%.4f msr=%.4f",
                batch_idx,
                len(loader),
                loss.item(),
                sop_loss.item(),
                msr_loss.item(),
            )

    n = max(n_batches, 1)
    result = {
        "loss": total_loss / n,
        "sop_loss": total_sop / n,
        "msr_loss": total_msr / n,
    }
    logger.info(
        "Pretrain %s epoch complete | loss=%.4f sop=%.4f msr=%.4f",
        mode,
        result["loss"],
        result["sop_loss"],
        result["msr_loss"],
    )
    return result
