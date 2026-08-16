from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import click
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kpi.models.STB import LectureSegmentationModel
from kpi.utils.stb_supervised import (
    load_processed_samples,
    probs_to_boundary_times_local_max,
    resolve_device,
)
from kpi.utils.text_label_dataset import (
    filter_internal_boundaries,
    map_frags_to_sentence_labels,
)


@dataclass(frozen=True)
class ModelSpec:
    name: str
    alias: str
    checkpoint_dir: Path
    threshold: float


def _load_model_from_best_dir(best_dir: Path, device: torch.device) -> LectureSegmentationModel:
    encoder_path = best_dir / "encoder.pt"
    transformer_path = best_dir / "transformer.pt"
    detector_path = best_dir / "detector.pt"

    missing = [
        str(path)
        for path in (encoder_path, transformer_path, detector_path)
        if not path.exists()
    ]
    if missing:
        raise click.BadParameter(
            f"Missing checkpoint files under {best_dir}: {missing}",
            param_hint="best_dir",
        )

    model = LectureSegmentationModel(
        encoder_checkpoint=encoder_path,
        transformer_checkpoint=transformer_path,
        detector_checkpoint=detector_path,
    )
    model = model.to(device)
    model.eval()
    return model


def _sentence_labels_from_boundaries(
    *,
    sentence_ends: list[float],
    duration: float,
    boundaries: list[float],
    sentence_count: int,
) -> list[int]:
    if sentence_count <= 0:
        return []

    capped_ends = sentence_ends[:sentence_count]
    if not capped_ends:
        return [0] * sentence_count

    cleaned = filter_internal_boundaries(boundaries, duration=duration)
    labels = map_frags_to_sentence_labels(capped_ends, cleaned)
    if len(labels) < sentence_count:
        labels.extend([0] * (sentence_count - len(labels)))
    return labels[:sentence_count]


def _predict_sentence_labels(
    *,
    model: LectureSegmentationModel,
    samples: list[dict[str, Any]],
    device: torch.device,
    batch_size: int,
    threshold: float,
    local_max_k: int,
) -> list[list[int]]:
    all_labels: list[list[int]] = [[0] * len(list(sample.get("text", []))) for sample in samples]

    with torch.no_grad():
        for start in range(0, len(samples), batch_size):
            batch_samples = samples[start : start + batch_size]
            batch_text = [list(sample.get("text", [])) for sample in batch_samples]
            lengths = torch.tensor(
                [len(text) for text in batch_text],
                dtype=torch.long,
                device=device,
            )

            logits = model(raw_text=batch_text, lengths=lengths)
            probs = torch.sigmoid(logits[..., 0]).detach().cpu()

            for local_idx, sample in enumerate(batch_samples):
                global_idx = start + local_idx
                sentence_count = len(batch_text[local_idx])
                sentence_ends = [float(x) for x in sample.get("sentence_ends", [])]
                duration = float(sample.get("duration", 0.0))
                cur_len = min(sentence_count, len(sentence_ends))

                if cur_len == 0:
                    all_labels[global_idx] = [0] * sentence_count
                    continue

                cur_probs = [float(probs[local_idx, sent_idx].item()) for sent_idx in range(cur_len)]
                pred_times = probs_to_boundary_times_local_max(
                    probs=cur_probs,
                    sentence_ends=sentence_ends[:cur_len],
                    threshold=threshold,
                    k=local_max_k,
                )

                all_labels[global_idx] = _sentence_labels_from_boundaries(
                    sentence_ends=sentence_ends,
                    duration=duration,
                    boundaries=[float(x) for x in pred_times],
                    sentence_count=sentence_count,
                )

    return all_labels


def _write_sample_documents(
    *,
    output_dir: Path,
    samples: list[dict[str, Any]],
    labels_by_alias: dict[str, list[list[int]]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    ordered_aliases = sorted(labels_by_alias.keys())

    for sample_idx, sample in enumerate(samples):
        sample_id = sample.get("id")
        sample_name = str(sample_id) if sample_id is not None else f"sample_{sample_idx:04d}"
        lines: list[str] = []

        sentences = [str(x) for x in sample.get("text", [])]
        for sent_idx, sentence in enumerate(sentences):
            yes_aliases: list[str] = []
            for alias in ordered_aliases:
                per_sample = labels_by_alias[alias][sample_idx]
                if sent_idx < len(per_sample) and int(per_sample[sent_idx]) == 1:
                    yes_aliases.append(alias)

            if yes_aliases:
                lines.append(f"{sentence} [boundary: {', '.join(yes_aliases)}]")
            else:
                lines.append(sentence)

        file_path = output_dir / f"{sample_name}.txt"
        file_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


@click.command()
@click.option(
    "--test_dataset_path",
    type=str,
    default="processed_dataset/mitfld_split/mitfld_processed_test.pt",
    show_default=True,
    help="Path to processed MITFLD test .pt file.",
)
@click.option(
    "--supervised_best_dir",
    type=str,
    default="artifacts/supervise_only/best",
    show_default=True,
    help="Directory with encoder.pt/transformer.pt/detector.pt for supervised-only model.",
)
@click.option(
    "--pretrained_best_dir",
    type=str,
    default="artifacts/pretrained_mix_6layer_BiLSTM/best",
    show_default=True,
    help="Directory with encoder.pt/transformer.pt/detector.pt for pretrained-mix model.",
)
@click.option(
    "--supervised_threshold",
    type=float,
    default=0.8,
    show_default=True,
    help="Decision threshold for supervised-only model.",
)
@click.option(
    "--pretrained_threshold",
    type=float,
    default=0.75,
    show_default=True,
    help="Decision threshold for pretrained-mix model.",
)
@click.option(
    "--local_max_k",
    type=int,
    default=5,
    show_default=True,
    help="Local-max window radius k.",
)
@click.option("--batch_size", type=int, default=4, show_default=True)
@click.option("--device", type=str, default="auto", show_default=True)
@click.option(
    "--max_samples",
    type=int,
    default=0,
    show_default=True,
    help="If > 0, only process the first N test samples. 0 means all samples.",
)
@click.option(
    "--output_dir",
    type=str,
    default="artifacts/mitfld_llm_judging_inputs",
    show_default=True,
)
def main(
    test_dataset_path: str,
    supervised_best_dir: str,
    pretrained_best_dir: str,
    supervised_threshold: float,
    pretrained_threshold: float,
    local_max_k: int,
    batch_size: int,
    device: str,
    max_samples: int,
    output_dir: str,
) -> None:
    if batch_size <= 0:
        raise click.BadParameter("batch_size must be > 0", param_hint="batch_size")
    if local_max_k < 0:
        raise click.BadParameter("local_max_k must be >= 0", param_hint="local_max_k")
    for value, name in (
        (supervised_threshold, "supervised_threshold"),
        (pretrained_threshold, "pretrained_threshold"),
    ):
        if not 0.0 <= value <= 1.0:
            raise click.BadParameter(f"{name} must be in [0, 1]", param_hint=name)
    if max_samples < 0:
        raise click.BadParameter("max_samples must be >= 0", param_hint="max_samples")

    samples = load_processed_samples(test_dataset_path)
    if not samples:
        raise click.BadParameter("No samples found in test dataset", param_hint="test_dataset_path")
    if max_samples > 0:
        samples = samples[:max_samples]

    model_specs = [
        ModelSpec(
            name="supervise_only",
            alias="B",
            checkpoint_dir=Path(supervised_best_dir),
            threshold=supervised_threshold,
        ),
        ModelSpec(
            name="pretrained_mix_6layer_BiLSTM",
            alias="C",
            checkpoint_dir=Path(pretrained_best_dir),
            threshold=pretrained_threshold,
        ),
    ]

    model_device = resolve_device(device)

    labels_by_alias: dict[str, list[list[int]]] = {
        "A": [],
        "B": [],
        "C": [],
    }

    for sample in samples:
        sentence_ends = [float(x) for x in sample.get("sentence_ends", [])]
        sentence_count = len(list(sample.get("text", [])))
        duration = float(sample.get("duration", 0.0))
        gt_boundaries = [float(x) for x in sample.get("frags", [])]
        labels_by_alias["A"].append(
            _sentence_labels_from_boundaries(
                sentence_ends=sentence_ends,
                duration=duration,
                boundaries=gt_boundaries,
                sentence_count=sentence_count,
            )
        )

    for spec in model_specs:
        model = _load_model_from_best_dir(spec.checkpoint_dir, model_device)
        labels_by_alias[spec.alias] = _predict_sentence_labels(
            model=model,
            samples=samples,
            device=model_device,
            batch_size=batch_size,
            threshold=spec.threshold,
            local_max_k=local_max_k,
        )

    output_root = Path(output_dir)
    docs_dir = output_root / "docs"
    _write_sample_documents(
        output_dir=docs_dir,
        samples=samples,
        labels_by_alias=labels_by_alias,
    )

    annotator_mapping = {
        "A": {
            "type": "ground_truth",
            "source": "MITFLD processed test labels",
        },
        "B": {
            "type": "model",
            "name": model_specs[0].name,
            "checkpoint_dir": str(model_specs[0].checkpoint_dir),
            "threshold": model_specs[0].threshold,
        },
        "C": {
            "type": "model",
            "name": model_specs[1].name,
            "checkpoint_dir": str(model_specs[1].checkpoint_dir),
            "threshold": model_specs[1].threshold,
        },
    }

    config = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "task": "Prepare MITFLD test sentence-level judging docs",
        "boundary_semantics": "Sentence is a boundary point when label=1",
        "dataset": {
            "test_dataset_path": test_dataset_path,
            "sample_count": len(samples),
            "max_samples": max_samples,
        },
        "prediction": {
            "local_max_k": local_max_k,
            "batch_size": batch_size,
            "device": str(model_device),
        },
        "annotator_mapping": annotator_mapping,
        "output": {
            "output_dir": str(output_root),
            "docs_dir": str(docs_dir),
            "doc_format": "One sentence per line; append [boundary: A, B, ...] only when boundary exists.",
            "include_timestamps": False,
            "include_header_metadata": False,
        },
    }

    output_root.mkdir(parents=True, exist_ok=True)
    config_path = output_root / "run_config.json"
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"Saved {len(samples)} sample documents to: {docs_dir}")
    print(f"Saved run config to: {config_path}")


if __name__ == "__main__":
    main()
