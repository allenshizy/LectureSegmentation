import random
import sys
from pathlib import Path
from typing import Any

import click
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kpi.datasets.mitfld import MITFLD
from kpi.models.STB import LectureSegmentationModel
from kpi.utils import (
    plot_probs_by_sentence,
    plot_probs_by_time,
)
from kpi.utils.lstm_predict import (
    format_fragments,
    load_video_ids,
    parse_requested_tokens,
)
from kpi.utils.logconfig import setup_logging
from kpi.utils.processed_split import partition_processed_paths
from kpi.utils.text_label_dataset import filter_internal_boundaries
from kpi.utils.stb_supervised import (
    load_processed_samples,
    probs_to_boundary_times_local_max,
    resolve_device,
    validate_split_ratios,
)


def _boundary_times_to_sentence_indices(
    sentence_ends: list[float],
    boundary_times: list[float],
    max_len: int,
) -> list[int]:
    indices: set[int] = set()
    capped_ends = sentence_ends[:max_len]
    for boundary in boundary_times:
        boundary_t = float(boundary)
        for sent_idx, end_t in enumerate(capped_ends):
            if boundary_t <= float(end_t):
                indices.add(sent_idx)
                break
    return sorted(indices)


def _group_boundary_times_by_sentence(
    sentence_ends: list[float],
    boundary_times: list[float],
    max_len: int,
) -> dict[int, list[float]]:
    grouped: dict[int, list[float]] = {}
    capped_ends = sentence_ends[:max_len]
    for boundary in boundary_times:
        boundary_t = float(boundary)
        for sent_idx, end_t in enumerate(capped_ends):
            if boundary_t <= float(end_t):
                grouped.setdefault(sent_idx, []).append(boundary_t)
                break
    return grouped


def _format_sentence_document(
    sample_id: str,
    split_name: str,
    sentences: list[str],
    pred_times: list[float],
    gt_times: list[float],
    sentence_ends: list[float],
    max_len: int,
) -> str:
    pred_by_sentence = _group_boundary_times_by_sentence(sentence_ends, pred_times, max_len)
    gt_by_sentence = _group_boundary_times_by_sentence(sentence_ends, gt_times, max_len)

    lines = [f"# {sample_id}", f"split: {split_name}", ""]
    for idx, sentence in enumerate(sentences[:max_len], start=1):
        parts = [f"{idx}. {sentence}"]
        pred_label = ", ".join(f"P: {time:.3f}s" for time in pred_by_sentence.get(idx - 1, []))
        gt_label = ", ".join(f"G: {time:.3f}s" for time in gt_by_sentence.get(idx - 1, []))
        if pred_label:
            parts.append(f"[{pred_label}]")
        if gt_label:
            parts.append(f"[{gt_label}]")
        lines.append(" ".join(parts))

    return "\n".join(lines).rstrip() + "\n"


def _build_selected_samples(
    dataset_path: str | None,
    processed_samples: list[dict[str, Any]] | None,
    selected_items: list[tuple[int, str]],
    all_video_ids: list[str],
) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    if processed_samples is not None:
        for idx, split_name in selected_items:
            sample = processed_samples[idx]
            sentence_times = [float(x) for x in sample.get("sentence_ends", [])]
            samples.append(
                {
                    "id": all_video_ids[idx],
                    "index": idx,
                    "split": split_name,
                    "text": list(sample.get("text", [])),
                    "sentence_times": sentence_times,
                    "duration": float(sample.get("duration", 0.0)),
                    "gt_times": [float(x) for x in sample.get("frags", [])],
                    "video_path": None,
                }
            )
        return samples

    dataset = MITFLD(dataset_path)
    for idx, split_name in selected_items:
        video = dataset.videos[idx]
        samples.append(
            {
                "id": all_video_ids[idx],
                "index": idx,
                "split": split_name,
                "text": [segment.text for segment in video.srt],
                "sentence_times": [float(segment.end) for segment in video.srt],
                "duration": float(video.duration),
                "gt_times": [float(x) for x in dataset.frags[idx]],
                "video_path": dataset.video_fns[idx],
            }
        )
    return samples


def _split_global_indices(
    total_size: int,
    ratios: list[float],
    seed: int,
) -> dict[str, list[int]]:
    rng = random.Random(seed)
    indices = list(range(total_size))
    rng.shuffle(indices)

    train_end = int(ratios[0] * len(indices))
    val_end = train_end + int(ratios[1] * len(indices))
    return {
        "train": indices[:train_end],
        "val": indices[train_end:val_end],
        "test": indices[val_end:],
    }


def _resolve_manual_indices(
    requested_tokens: list[str],
    all_video_ids: list[str],
) -> list[int]:
    id_to_index = {vid: idx for idx, vid in enumerate(all_video_ids)}
    selected: list[int] = []
    used: set[int] = set()

    for token in requested_tokens:
        idx = None
        if token in id_to_index:
            idx = id_to_index[token]
        elif token.isdigit() and 0 <= int(token) < len(all_video_ids):
            idx = int(token)
        else:
            raise click.BadParameter(
                f"Unknown video token '{token}'. Provide a valid video ID or index.",
                param_hint="video",
            )

        if idx not in used:
            selected.append(idx)
            used.add(idx)
    return selected


@click.command()
@click.option(
    "--dataset_path",
    required=False,
    type=str,
    help="Path to the MITFLD dataset root directory.",
)
@click.option(
    "--processed_dataset_path",
    "processed_dataset_paths",
    required=False,
    multiple=True,
    type=str,
    help=(
        "Path to a processed text-label dataset .pt file. Repeat to provide "
        "pre-split train/validation/test files."
    ),
)
@click.option(
    "--checkpoint_path",
    required=True,
    type=str,
    help="Path to a saved STB LectureSegmentationModel checkpoint (.pt).",
)
@click.option(
    "--split_seed",
    type=int,
    default=2024,
    show_default=True,
    help="Seed used for train/val/test split shuffling.",
)
@click.option("--train_ratio", type=float, default=0.7, show_default=True)
@click.option("--val_ratio", type=float, default=0.15, show_default=True)
@click.option("--test_ratio", type=float, default=0.15, show_default=True)
@click.option("--num_train_samples", type=int, default=1, show_default=True)
@click.option("--num_val_samples", type=int, default=1, show_default=True)
@click.option("--num_test_samples", type=int, default=1, show_default=True)
@click.option(
    "--video",
    "video_tokens",
    multiple=True,
    help=(
        "Optional explicit sample selectors (repeatable or comma-separated). "
        "If provided, split-based sampling is skipped. "
        "For MITFLD, each value can be a video ID from video_id_list.txt or a 0-based index. "
        "For processed dataset, each value can be sample_<index> or a 0-based index."
    ),
)
@click.option("--sample_seed", type=int, default=2024, show_default=True, help="Seed used for sampling inside each split.")
@click.option("--random_sample_seed/--no-random_sample_seed", default=False, show_default=True, help="Use a fresh random sampling seed on each run.")
@click.option("--decision_threshold", type=float, default=0.5, help="Boundary decision threshold in [0, 1].")
@click.option("--local_max_k", type=int, default=3, show_default=True, help="Local-maximum window radius k.")
@click.option("--device", type=str, default="auto", help="Torch device string or 'auto'.")
@click.option("--batch_size", type=int, default=4)
@click.option("--loglevel", type=str, default="DID", help="Logging level (e.g., DEBUG, INFO, WARNING, ERROR, CRITICAL).")
@click.option("--save_prob_plots/--no-save_prob_plots", default=False, show_default=True)
@click.option("--output_dir", type=str, default="results/stb_outputs", show_default=True)
@click.option("--save_sentence_doc/--no-save_sentence_doc", default=False, show_default=True)

def run_demo_stb_predict_experiment(
    dataset_path: str | None,
    processed_dataset_paths: tuple[str, ...],
    checkpoint_path: str,
    split_seed: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    num_train_samples: int,
    num_val_samples: int,
    num_test_samples: int,
    video_tokens: tuple[str, ...],
    sample_seed: int,
    random_sample_seed: bool,
    decision_threshold: float,
    local_max_k: int,
    device: str,
    batch_size: int,
    loglevel: str,
    save_prob_plots: bool,
    output_dir: str,
    save_sentence_doc: bool,
):
    using_processed_dataset = len(processed_dataset_paths) > 0

    if bool(dataset_path) == using_processed_dataset:
        raise click.BadParameter(
            "Provide exactly one of --dataset_path or --processed_dataset_path"
        )
    if batch_size <= 0:
        raise click.BadParameter("batch_size must be > 0", param_hint="batch_size")
    if num_train_samples < 0:
        raise click.BadParameter("num_train_samples must be >= 0", param_hint="num_train_samples")
    if num_val_samples < 0:
        raise click.BadParameter("num_val_samples must be >= 0", param_hint="num_val_samples")
    if num_test_samples < 0:
        raise click.BadParameter("num_test_samples must be >= 0", param_hint="num_test_samples")
    if (num_train_samples + num_val_samples + num_test_samples) <= 0:
        raise click.BadParameter(
            "At least one of num_train_samples/num_val_samples/num_test_samples must be > 0"
        )
    if not 0.0 <= decision_threshold <= 1.0:
        raise click.BadParameter(
            "decision_threshold must be in [0, 1]",
            param_hint="decision_threshold",
        )
    if local_max_k < 0:
        raise click.BadParameter("local_max_k must be >= 0", param_hint="local_max_k")

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    
    setup_logging(loglevel, output_dir)

    print("Running STB segmentation prediction experiment with loaded checkpoint")

    if not using_processed_dataset:
        validate_split_ratios(train_ratio, val_ratio, test_ratio)

    processed_samples = None
    pre_split_indices: dict[str, list[int]] | None = None
    if using_processed_dataset:
        train_paths, val_paths, test_paths = partition_processed_paths(list(processed_dataset_paths))
        split_paths = {
            "train": train_paths,
            "val": val_paths,
            "test": test_paths,
        }

        processed_samples = []
        pre_split_indices = {"train": [], "val": [], "test": []}
        for split_name in ("train", "val", "test"):
            for path in split_paths[split_name]:
                loaded = load_processed_samples(path)
                for sample in loaded:
                    pre_split_indices[split_name].append(len(processed_samples))
                    processed_samples.append(sample)

        if not processed_samples:
            raise click.BadParameter(
                "No samples found in provided --processed_dataset_path files",
                param_hint="processed_dataset_path",
            )

        all_video_ids = [f"sample_{idx}" for idx in range(len(processed_samples))]
    else:
        all_video_ids = load_video_ids(dataset_path)

    requested_tokens = parse_requested_tokens(video_tokens)
    if requested_tokens:
        selected_indices = _resolve_manual_indices(requested_tokens, all_video_ids)
        selected_items = [(idx, "manual") for idx in selected_indices]
    else:
        if using_processed_dataset:
            split_indices = {
                "train": list((pre_split_indices or {}).get("train", [])),
                "val": list((pre_split_indices or {}).get("val", [])),
                "test": list((pre_split_indices or {}).get("test", [])),
            }
        else:
            split_indices = _split_global_indices(
                total_size=len(all_video_ids),
                ratios=[train_ratio, val_ratio, test_ratio],
                seed=split_seed,
            )

        if num_train_samples > len(split_indices["train"]):
            raise click.BadParameter(
                f"num_train_samples ({num_train_samples}) exceeds train split size ({len(split_indices['train'])})",
                param_hint="num_train_samples",
            )
        if num_val_samples > len(split_indices["val"]):
            raise click.BadParameter(
                f"num_val_samples ({num_val_samples}) exceeds val split size ({len(split_indices['val'])})",
                param_hint="num_val_samples",
            )
        if num_test_samples > len(split_indices["test"]):
            raise click.BadParameter(
                f"num_test_samples ({num_test_samples}) exceeds test split size ({len(split_indices['test'])})",
                param_hint="num_test_samples",
            )

        effective_sample_seed = sample_seed
        if random_sample_seed:
            effective_sample_seed = random.SystemRandom().randint(0, 2**31 - 1)
            print(f"Using random sample_seed={effective_sample_seed}")

        sample_rng = random.Random(effective_sample_seed)
        sampled_splits: dict[str, list[int]] = {}
        for split_name in ("train", "val", "test"):
            cur = list(split_indices[split_name])
            sample_rng.shuffle(cur)
            sampled_splits[split_name] = cur

        selected_items = []
        selected_items.extend((idx, "train") for idx in sampled_splits["train"][:num_train_samples])
        selected_items.extend((idx, "val") for idx in sampled_splits["val"][:num_val_samples])
        selected_items.extend((idx, "test") for idx in sampled_splits["test"][:num_test_samples])

    model_device = resolve_device(device)
    model = LectureSegmentationModel.load_checkpoint(
        checkpoint_path,
        map_location="cpu",
    )
    model = model.to(model_device)
    model.eval()

    selected_samples = _build_selected_samples(
        dataset_path=dataset_path,
        processed_samples=processed_samples,
        selected_items=selected_items,
        all_video_ids=all_video_ids,
    )

    results: list[dict[str, Any]] = []
    with torch.no_grad():
        for start in range(0, len(selected_samples), batch_size):
            batch_samples = selected_samples[start : start + batch_size]
            batch_text = [sample["text"] for sample in batch_samples]
            lengths = torch.tensor(
                [len(cur) for cur in batch_text],
                dtype=torch.long,
                device=model_device,
            )

            logits = model(raw_text=batch_text, lengths=lengths)
            probs = torch.sigmoid(logits[..., 0]).detach().cpu()

            for local_idx, sample in enumerate(batch_samples):
                sentence_times = [float(x) for x in sample["sentence_times"]]
                duration = float(sample["duration"])
                gt_times = filter_internal_boundaries(
                    [float(x) for x in sample["gt_times"]],
                    duration=duration,
                )
                cur_len = min(int(lengths[local_idx].item()), len(sentence_times))
                cur_probs = [float(probs[local_idx, sent_idx].item()) for sent_idx in range(cur_len)]

                pred_times = filter_internal_boundaries(
                    probs_to_boundary_times_local_max(
                    probs=cur_probs,
                    sentence_ends=sentence_times[:cur_len],
                    threshold=decision_threshold,
                    k=local_max_k,
                ),
                    duration=duration,
                )
                pred_boundaries = pred_times
                pred_boundaries = sorted({0.0, *pred_boundaries, duration})

                if save_prob_plots:
                    cur_id = str(sample["id"])
                    gt_sentence_indices = _boundary_times_to_sentence_indices(
                        sentence_ends=sentence_times,
                        boundary_times=gt_times,
                        max_len=cur_len,
                    )
                    sentence_plot_path = output_root / f"{cur_id}_sentence.png"
                    time_plot_path = output_root / f"{cur_id}_time.png"

                    plot_probs_by_sentence(
                        probs=cur_probs,
                        gt_sentence_indices=gt_sentence_indices,
                        save_path=sentence_plot_path,
                        title=f"{cur_id} - probs vs sentence index",
                        threshold=decision_threshold,
                    )
                    plot_probs_by_time(
                        probs=cur_probs,
                        sentence_times=sentence_times[:cur_len],
                        gt_times=gt_times,
                        save_path=time_plot_path,
                        duration=duration,
                        title=f"{cur_id} - probs vs time",
                        threshold=decision_threshold,
                    )

                if save_sentence_doc:
                    cur_id = str(sample["id"])
                    doc_path = output_root / f"{cur_id}.md"
                    doc_content = _format_sentence_document(
                        sample_id=cur_id,
                        split_name=str(sample["split"]),
                        sentences=list(sample["text"]),
                        pred_times=pred_times,
                        gt_times=gt_times,
                        sentence_ends=sentence_times[:cur_len],
                        max_len=cur_len,
                    )
                    doc_path.write_text(doc_content, encoding="utf-8")

                results.append(
                    {
                        "id": sample["id"],
                        "index": sample["index"],
                        "split": sample["split"],
                        "video_path": sample["video_path"],
                        "prediction": pred_boundaries,
                        "ground_truth": gt_times,
                    }
                )

    print("\nPrediction results:")
    for order, result in enumerate(results, start=1):
        print(f"[{order}] id={result['id']}, index={result['index']}, split={result['split']}")
        if result["video_path"] is not None:
            print(f"  video_path: {result['video_path']}")
        print(f"  prediction: {format_fragments(result['prediction'])}")
        print(f"  ground_truth: {format_fragments(result['ground_truth'])}")

    if save_prob_plots:
        print(f"Saved probability plots to: {output_root}")
    if save_sentence_doc:
        print(f"Saved sentence documents to: {output_root}")


if __name__ == "__main__":
    run_demo_stb_predict_experiment()