"""Training-loop helpers for supervised STB segmentation."""

from __future__ import annotations

import logging

import torch
from torch import nn
from torch.utils.data import DataLoader

from kpi.models.STB import LectureSegmentationModel
from kpi.utils import TextLabelSequenceDataset


logger = logging.getLogger(__name__)


def compute_pos_weight(train_dataset: TextLabelSequenceDataset, clamp_max: float = 100.0) -> float:
    """Estimate BCE positive-class weight from sentence labels."""

    pos = 0
    neg = 0
    for idx in range(len(train_dataset)):
        labels: torch.Tensor = train_dataset[idx]["label"]
        cur_pos = int(labels.sum().item())
        cur_total = int(labels.numel())
        pos += cur_pos
        neg += cur_total - cur_pos

    if pos <= 0:
        logger.warning("No positive labels found in train dataset, fallback pos_weight=1.0")
        return 1.0
    pos_weight = float(min(max(neg / pos, 1.0), clamp_max))
    logger.info("Computed pos_weight=%.4f from pos=%d neg=%d", pos_weight, pos, neg)
    return pos_weight


def pad_labels(
    labels: list[torch.Tensor],
    max_len: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad variable-length label tensors and emit a valid-position mask."""

    batch_size = len(labels)
    target = torch.zeros((batch_size, max_len, 1), dtype=torch.float32, device=device)
    valid_mask = torch.zeros((batch_size, max_len, 1), dtype=torch.float32, device=device)
    for i, label in enumerate(labels):
        cur = label.to(device=device, dtype=torch.float32)
        cur_len = min(int(cur.size(0)), max_len)
        if cur_len == 0:
            continue
        target[i, :cur_len, 0] = cur[:cur_len]
        valid_mask[i, :cur_len, 0] = 1.0
    logger.debug("Padded labels to shape=%s", tuple(target.shape))
    return target, valid_mask


def run_epoch(
    model: LectureSegmentationModel,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    grad_clip_norm: float | None = None,
) -> dict[str, float]:
    """Run one training or validation epoch and return average masked BCE loss."""

    is_train = optimizer is not None
    model.train(is_train)
    mode = "train" if is_train else "eval"
    total_params = sum(param.numel() for param in model.parameters())
    trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    logger.info(
        "%s epoch trainable params: %d / %d",
        mode,
        trainable_params,
        total_params,
    )
    logger.debug("Starting %s epoch over %d batches", mode, len(loader))

    total_loss = 0.0
    total_valid = 0.0

    for batch_idx, batch in enumerate(loader, start=1):
        lengths = batch["lengths"].to(device)
        logits = model(raw_text=batch["text"], lengths=lengths)

        target, valid_mask = pad_labels(batch["label"], logits.size(1), device)
        loss_elem = criterion(logits, target)
        loss = (loss_elem * valid_mask).sum() / valid_mask.sum().clamp_min(1.0)

        if is_train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip_norm is not None and grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()

        batch_valid = float(valid_mask.sum().item())
        total_loss += float(loss.item()) * batch_valid
        total_valid += batch_valid
        logger.debug(
            "%s batch=%d loss=%.6f valid_tokens=%.0f",
            mode,
            batch_idx,
            float(loss.item()),
            batch_valid,
        )

    avg_loss = total_loss / total_valid if total_valid > 0 else 0.0
    logger.info(
        "%s epoch done: loss=%.6f",
        mode,
        avg_loss,
    )
    return {"loss": avg_loss}
