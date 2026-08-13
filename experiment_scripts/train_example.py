"""Minimal usage examples for the STB lecture segmentation framework."""

from __future__ import annotations

from pathlib import Path
import logging

import torch
from torch import nn

from kpi.models.STB import BoundaryDetector, GlobalTransformer, LectureSegmentationModel, SentenceEncoder


def setup_logging() -> Path:
    """Configure root logging to both console and a local .log file."""

    log_dir = Path(__file__).resolve().parent
    log_path = log_dir / "train.log"
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.handlers.clear()

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    root_logger.info("Logging initialized. Log file: %s", log_path)
    return log_path

def example_1_end_to_end_training() -> None:
    model = LectureSegmentationModel() 
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    raw_text = [["The introduction starts here.", "A key idea appears.", "The topic changes."]]
    labels = torch.tensor([[[0.0], [1.0], [0.0]]])

    logits = model(raw_text=raw_text)
    loss = criterion(logits, labels)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)


def example_2_frozen_encoder_training() -> None:
    model = LectureSegmentationModel().freeze_encoder()
    optimizer = torch.optim.AdamW(
        list(model.transformer.parameters()) + list(model.detector.parameters()),
        lr=1e-4,
    )

    sentence_embeddings = torch.randn(2, 5, 384)
    labels = torch.zeros(2, 5, 1)

    logits = model(sentence_embeddings=sentence_embeddings)
    loss = nn.BCEWithLogitsLoss()(logits, labels)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)


def example_3_cached_embedding_training() -> None:
    cache_dir = Path("artifacts/stb_examples")
    cache_dir.mkdir(parents=True, exist_ok=True)

    encoder = SentenceEncoder()
    cached_embeddings, lengths = encoder([["Cached sentence one.", "Cached sentence two."]], return_lengths=True)
    embedding_path = cache_dir / "embeddings.pt"
    encoder.save_features(cached_embeddings, embedding_path, lengths=lengths)

    cached = SentenceEncoder.load_features(embedding_path)
    model = LectureSegmentationModel()
    logits = model(sentence_embeddings=cached["features"], lengths=cached["lengths"])
    labels = torch.zeros_like(logits)
    loss = nn.BCEWithLogitsLoss()(logits, labels)
    loss.backward()


def example_4_detector_only_training() -> None:
    cache_dir = Path("artifacts/stb_examples")
    cache_dir.mkdir(parents=True, exist_ok=True)

    detector = BoundaryDetector()
    transformer_features = torch.randn(2, 4, 384)
    feature_path = cache_dir / "transformer_features.pt"
    detector.save_features(transformer_features, feature_path, lengths=[4, 3])

    cached = BoundaryDetector.load_features(feature_path)
    optimizer = torch.optim.AdamW(detector.parameters(), lr=1e-4)
    logits = detector(cached["features"], lengths=cached["lengths"])
    labels = torch.zeros_like(logits)
    loss = nn.BCEWithLogitsLoss()(logits, labels)
    loss.backward()
    optimizer.step()


def example_5_saving_and_loading() -> None:
    cache_dir = Path("artifacts/stb_examples")
    cache_dir.mkdir(parents=True, exist_ok=True)

    model = LectureSegmentationModel()
    model_path = cache_dir / "lecture_segmentation.pt"
    model.save_checkpoint(model_path)
    _ = LectureSegmentationModel.load_checkpoint(model_path)

    encoder_path = cache_dir / "sentence_encoder.pt"
    transformer_path = cache_dir / "global_transformer.pt"
    detector_path = cache_dir / "boundary_detector.pt"
    model.encoder.save_checkpoint(encoder_path)
    model.transformer.save_checkpoint(transformer_path)
    model.detector.save_checkpoint(detector_path)
    _ = SentenceEncoder.load_checkpoint(encoder_path)
    _ = GlobalTransformer.load_checkpoint(transformer_path)
    _ = BoundaryDetector.load_checkpoint(detector_path)


if __name__ == "__main__":
    setup_logging()
    example_1_end_to_end_training()
    example_2_frozen_encoder_training()
    example_3_cached_embedding_training()
    example_4_detector_only_training()
    example_5_saving_and_loading()