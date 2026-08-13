from __future__ import annotations

import logging

import torch

from app.config import STBConfig
from app.pipeline.asr import Segment
from kpi.models.STB import LectureSegmentationModel
from kpi.utils.stb_supervised import probs_to_boundary_times_local_max, resolve_device

logger = logging.getLogger(__name__)


class StbSegmenter:
    """Wraps the pretrained LectureSegmentationModel for boundary detection over ASR segments."""

    def __init__(self, config: STBConfig | None = None) -> None:
        self.config = config or STBConfig()
        self._model: LectureSegmentationModel | None = None
        self._device: torch.device | None = None

    def _load_model(self) -> LectureSegmentationModel:
        if self._model is not None:
            return self._model
        device = resolve_device(self.config.device)
        logger.info("Loading STB model checkpoints onto device=%s", device)
        model = LectureSegmentationModel(
            # encoder left as default: un-finetuned SBERT, fetched/cached from HuggingFace.
            transformer_checkpoint=self.config.transformer_checkpoint,
            detector_checkpoint=self.config.detector_checkpoint,
        )
        model = model.to(device)
        model.eval()
        self._model = model
        self._device = device
        return model

    @torch.no_grad()
    def predict_boundaries(self, segments: list[Segment]) -> list[float]:
        """Return boundary timestamps (segment end times) marking chapter breaks."""

        if not segments:
            return []

        model = self._load_model()
        sentences = [seg.text for seg in segments]
        sentence_ends = [seg.end for seg in segments]

        logits = model(raw_text=[sentences])
        probs = torch.sigmoid(logits[0, :, 0]).detach().cpu().tolist()

        boundary_times = probs_to_boundary_times_local_max(
            probs=probs,
            sentence_ends=sentence_ends,
            threshold=self.config.threshold,
            k=self.config.local_max_window,
        )
        logger.info("Predicted %d boundaries out of %d segments", len(boundary_times), len(segments))
        return boundary_times
