"""Training-loop helpers for STB pre-training (SOP + MSR objectives)."""

from __future__ import annotations

import logging

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

    total_loss = 0.0
    total_sop = 0.0
    total_msr = 0.0
    n_batches = 0

    for batch_idx, batch in enumerate(loader, start=1):
        lengths: torch.Tensor = batch["lengths"].to(device)

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
            loss.backward()
            if grad_clip_norm is not None:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()

        total_loss += loss.item()
        total_sop += sop_loss.item()
        total_msr += msr_loss.item()
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
