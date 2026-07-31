"""Evaluation helpers for supervised STB segmentation."""

from __future__ import annotations

import logging
from typing import Any

import torch
from torch.utils.data import DataLoader

from kpi.metrics.f1 import F1
from kpi.metrics.iou import IoU
from kpi.metrics.mof import MoF
from kpi.models.STB import LectureSegmentationModel
from kpi.utils.stb_supervised.predict import probs_to_boundary_times_local_max


logger = logging.getLogger(__name__)


def _score_predictions(
    all_preds: list[list[float]],
    all_gts: list[list[float]],
    f1_threshold: float,
    iou_fps: int,
    mof_fps: int,
) -> dict[str, float]:
    scores: dict[str, float] = {}
    metrics = [
        F1(threshold=f1_threshold),
        IoU(fps=iou_fps),
        MoF(fps=mof_fps),
    ]
    for metric in metrics:
        pred_copy = [pred[:] for pred in all_preds]
        gt_copy = [gt[:] for gt in all_gts]
        scores[str(metric)] = float(metric(pred_copy, gt_copy))
    return scores


def _collect_processed_prob_records(
    model: LectureSegmentationModel,
    loader: DataLoader,
    device: torch.device,
) -> list[tuple[list[float], list[float], float, list[float]]]:
    records: list[tuple[list[float], list[float], float, list[float]]] = []
    model.eval()
    logger.debug("Collecting processed probability records from %d batches", len(loader))

    with torch.no_grad():
        for batch in loader:
            lengths = batch["lengths"].to(device)
            logits = model(raw_text=batch["text"], lengths=lengths)
            probs = torch.sigmoid(logits[..., 0]).detach().cpu()

            for i, sentence_ends in enumerate(batch["sentence_ends"]):
                cur_len = min(int(lengths[i].item()), len(sentence_ends))
                cur_probs = [float(probs[i, sent_idx].item()) for sent_idx in range(cur_len)]
                cur_sentence_ends = [float(x) for x in sentence_ends[:cur_len]]
                cur_duration = float(batch["duration"][i])
                cur_gt = canonicalize_boundaries(batch["frags"][i], cur_duration)
                records.append((cur_probs, cur_sentence_ends, cur_duration, cur_gt))

    logger.debug("Collected %d processed probability records", len(records))
    return records


def _collect_raw_prob_records(
    model: LectureSegmentationModel,
    dataset_split: Any,
    loader: DataLoader,
    device: torch.device,
) -> list[tuple[list[float], list[float], float, list[float]]]:
    records: list[tuple[list[float], list[float], float, list[float]]] = []
    model.eval()
    offset = 0
    logger.debug("Collecting raw probability records from %d batches", len(loader))

    with torch.no_grad():
        for batch in loader:
            batch_size = len(batch["text"])
            batch_videos = dataset_split.videos[offset : offset + batch_size]
            lengths = batch["lengths"].to(device)
            logits = model(raw_text=batch["text"], lengths=lengths)
            probs = torch.sigmoid(logits[..., 0]).detach().cpu()

            for i, video in enumerate(batch_videos):
                cur_len = int(lengths[i].item())
                cur_probs = [float(probs[i, sent_idx].item()) for sent_idx in range(cur_len)]
                cur_sentence_ends = [float(video.srt[sent_idx].end) for sent_idx in range(cur_len)]
                cur_duration = float(video.duration)
                cur_gt = canonicalize_boundaries(dataset_split.frags[offset + i], cur_duration)
                records.append((cur_probs, cur_sentence_ends, cur_duration, cur_gt))

            offset += batch_size

            logger.debug("Collected %d raw probability records", len(records))
    return records


def _decode_records(
    records: list[tuple[list[float], list[float], float, list[float]]],
    boundary_threshold: float,
    local_max_k: int,
) -> tuple[list[list[float]], list[list[float]]]:
    all_preds: list[list[float]] = []
    all_gts: list[list[float]] = []

    for probs, sentence_ends, duration, gt in records:
        pred_times = probs_to_boundary_times_local_max(
            probs=probs,
            sentence_ends=sentence_ends,
            threshold=boundary_threshold,
            k=local_max_k,
        )
        all_preds.append(canonicalize_boundaries(pred_times, duration))
        all_gts.append(gt)

    logger.debug(
        "Decoded %d records with threshold=%.4f and local_max_k=%d",
        len(records),
        boundary_threshold,
        local_max_k,
    )
    return all_preds, all_gts


def _select_best_threshold_from_records(
    records: list[tuple[list[float], list[float], float, list[float]]],
    threshold_candidates: list[float],
    local_max_k: int,
    f1_threshold: float,
) -> tuple[float, float, list[dict[str, float]]]:
    if not threshold_candidates:
        raise ValueError("threshold_candidates must not be empty")

    logger.debug(
        "Running threshold sweep over %d candidates on %d records (local_max_k=%d, f1_threshold=%.4f)",
        len(threshold_candidates),
        len(records),
        local_max_k,
        f1_threshold,
    )

    metric = F1(threshold=f1_threshold)
    sweep: list[dict[str, float]] = []
    best_threshold = float(threshold_candidates[0])
    best_f1 = float("-inf")

    for threshold in threshold_candidates:
        all_preds, all_gts = _decode_records(
            records=records,
            boundary_threshold=float(threshold),
            local_max_k=local_max_k,
        )
        cur_f1 = float(metric([pred[:] for pred in all_preds], [gt[:] for gt in all_gts]))
        sweep.append({"threshold": float(threshold), "f1": cur_f1})
        logger.debug("Threshold %.4f -> val F1 %.6f", float(threshold), cur_f1)
        if cur_f1 > best_f1:
            best_f1 = cur_f1
            best_threshold = float(threshold)

    logger.debug(
        "Threshold sweep done: best_threshold=%.4f best_f1=%.6f",
        best_threshold,
        best_f1,
    )

    return best_threshold, best_f1, sweep


def select_best_boundary_threshold_processed(
    model: LectureSegmentationModel,
    loader: DataLoader,
    device: torch.device,
    threshold_candidates: list[float],
    local_max_k: int,
    f1_threshold: float,
) -> tuple[float, float, list[dict[str, float]]]:
    """Select best threshold on processed split using one model pass and offline threshold sweep."""

    logger.info("Selecting best boundary threshold on processed validation split")

    records = _collect_processed_prob_records(model=model, loader=loader, device=device)
    best_threshold, best_f1, sweep = _select_best_threshold_from_records(
        records=records,
        threshold_candidates=threshold_candidates,
        local_max_k=local_max_k,
        f1_threshold=f1_threshold,
    )
    logger.info(
        "Processed validation threshold selection done: threshold=%.4f f1=%.6f",
        best_threshold,
        best_f1,
    )
    return best_threshold, best_f1, sweep


def select_best_boundary_threshold(
    model: LectureSegmentationModel,
    dataset_split: Any,
    loader: DataLoader,
    device: torch.device,
    threshold_candidates: list[float],
    local_max_k: int,
    f1_threshold: float,
) -> tuple[float, float, list[dict[str, float]]]:
    """Select best threshold on raw split using one model pass and offline threshold sweep."""

    logger.debug("Selecting best boundary threshold on raw validation split")

    records = _collect_raw_prob_records(
        model=model,
        dataset_split=dataset_split,
        loader=loader,
        device=device,
    )
    best_threshold, best_f1, sweep = _select_best_threshold_from_records(
        records=records,
        threshold_candidates=threshold_candidates,
        local_max_k=local_max_k,
        f1_threshold=f1_threshold,
    )
    logger.debug(
        "Raw validation threshold selection done: threshold=%.4f f1=%.6f",
        best_threshold,
        best_f1,
    )
    return best_threshold, best_f1, sweep


def canonicalize_boundaries(boundaries: list[float], duration: float) -> list[float]:
    """Normalize boundaries to sorted unique interior points with [0, duration] caps."""

    internal = sorted({float(x) for x in boundaries if 0.0 < float(x) < float(duration)})
    return [0.0] + internal + [float(duration)]


def evaluate_boundary_metrics_processed(
    model: LectureSegmentationModel,
    loader: DataLoader,
    device: torch.device,
    boundary_threshold: float,
    local_max_k: int,
    f1_threshold: float,
    iou_fps: int,
    mof_fps: int,
) -> dict[str, float]:
    """Evaluate metrics using preprocessed batches with sentence_ends metadata."""
    records = _collect_processed_prob_records(model=model, loader=loader, device=device)
    all_preds, all_gts = _decode_records(
        records=records,
        boundary_threshold=boundary_threshold,
        local_max_k=local_max_k,
    )
    return _score_predictions(
        all_preds=all_preds,
        all_gts=all_gts,
        f1_threshold=f1_threshold,
        iou_fps=iou_fps,
        mof_fps=mof_fps,
    )


def decode_logits_to_boundaries(
    logits: torch.Tensor,
    lengths: torch.Tensor,
    videos: list[Any],
    threshold: float,
    local_max_k: int,
) -> list[list[float]]:
    """Decode sentence logits to boundary-time predictions per video."""

    logger.debug("Decoding logits to boundaries for batch_size=%d", len(videos))
    probs = torch.sigmoid(logits[..., 0]).detach().cpu()
    predictions: list[list[float]] = []

    for i, video in enumerate(videos):
        cur_len = int(lengths[i].item())
        cur_probs = [float(probs[i, sent_idx].item()) for sent_idx in range(cur_len)]
        pred_times = probs_to_boundary_times_local_max(
            probs=cur_probs,
            sentence_ends=[float(video.srt[sent_idx].end) for sent_idx in range(cur_len)],
            threshold=threshold,
            k=local_max_k,
        )
        predictions.append(canonicalize_boundaries(pred_times, float(video.duration)))
    logger.debug("Decoded %d boundary sequences", len(predictions))
    return predictions


def evaluate_boundary_metrics(
    model: LectureSegmentationModel,
    dataset_split: Any,
    loader: DataLoader,
    device: torch.device,
    boundary_threshold: float,
    local_max_k: int,
    f1_threshold: float,
    iou_fps: int,
    mof_fps: int,
) -> dict[str, float]:
    """Run test inference and compute F1/IoU/MoF on boundary times."""

    model.eval()
    logger.info(
        "Evaluating boundary metrics with threshold=%.3f, f1_threshold=%.3f",
        boundary_threshold,
        f1_threshold,
    )
    records = _collect_raw_prob_records(
        model=model,
        dataset_split=dataset_split,
        loader=loader,
        device=device,
    )
    all_preds, all_gts = _decode_records(
        records=records,
        boundary_threshold=boundary_threshold,
        local_max_k=local_max_k,
    )
    logger.debug("Collected predictions for %d videos", len(all_preds))

    scores = _score_predictions(
        all_preds=all_preds,
        all_gts=all_gts,
        f1_threshold=f1_threshold,
        iou_fps=iou_fps,
        mof_fps=mof_fps,
    )
    for metric_name, value in scores.items():
        logger.info("Metric %s=%.6f", metric_name, value)
    return scores
