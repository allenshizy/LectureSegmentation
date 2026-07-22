from __future__ import annotations

from collections.abc import Sequence
import logging
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from kpi.models.STB.base_module import BaseModule
from kpi.models.STB.bilstm_head import BoundaryDetector
from kpi.models.STB.global_transformer import GlobalTransformer
from kpi.models.STB.sentence_encoder import SentenceEncoder


logger = logging.getLogger(__name__)


class LectureSegmentationModel(BaseModule):
    """Unified lecture segmentation interface.

    Constructor:
        encoder, transformer, detector: Optional prebuilt submodules.
        encoder_checkpoint, transformer_checkpoint, detector_checkpoint:
            Optional paths to module checkpoints.
        encoder_kwargs, transformer_kwargs, detector_kwargs: Keyword arguments
            used to instantiate default submodules.

    Input shapes:
        raw_text: ``Sequence[Sequence[str]]`` or ``Sequence[str]``
        sentence_embeddings: ``[batch, sequence_length, 384]``
        transformer_features: ``[batch, sequence_length, 384]``

    Output shape:
        ``[batch, sequence_length, 1]`` boundary logits.

    Usage example:
        >>> model = LectureSegmentationModel()
        >>> logits = model(raw_text=[["One.", "Two."]])
        >>> model.save_checkpoint("lecture_segmentation.pt")
        >>> restored = LectureSegmentationModel.load_checkpoint("lecture_segmentation.pt")
    """

    def __init__(
        self,
        encoder: SentenceEncoder | None = None,
        transformer: GlobalTransformer | None = None,
        detector: BoundaryDetector | None = None,
        *,
        encoder_checkpoint: str | Path | None = None,
        transformer_checkpoint: str | Path | None = None,
        detector_checkpoint: str | Path | None = None,
        encoder_kwargs: dict[str, Any] | None = None,
        transformer_kwargs: dict[str, Any] | None = None,
        detector_kwargs: dict[str, Any] | None = None,
    ) -> None:
        encoder_kwargs = dict(encoder_kwargs or {})
        transformer_kwargs = dict(transformer_kwargs or {})
        detector_kwargs = dict(detector_kwargs or {})

        if encoder is None:
            encoder = (
                SentenceEncoder.load_checkpoint(encoder_checkpoint)
                if encoder_checkpoint is not None
                else SentenceEncoder(**encoder_kwargs)
            )
        if transformer is None:
            transformer = (
                GlobalTransformer.load_checkpoint(transformer_checkpoint)
                if transformer_checkpoint is not None
                else GlobalTransformer(**transformer_kwargs)
            )
        if detector is None:
            detector = (
                BoundaryDetector.load_checkpoint(detector_checkpoint)
                if detector_checkpoint is not None
                else BoundaryDetector(**detector_kwargs)
            )

        super().__init__(
            {
                "encoder": encoder.get_config(),
                "transformer": transformer.get_config(),
                "detector": detector.get_config(),
            }
        )
        self.encoder = encoder
        self.transformer = transformer
        self.detector = detector
        logger.info(
            "Initialized LectureSegmentationModel(encoder=%s, transformer=%s, detector=%s)",
            self.encoder.__class__.__name__,
            self.transformer.__class__.__name__,
            self.detector.__class__.__name__,
        )

    def get_config(self) -> dict[str, Any]:
        self.config = {
            "encoder": self.encoder.get_config(),
            "transformer": self.transformer.get_config(),
            "detector": self.detector.get_config(),
        }
        return dict(self.config)

    @staticmethod
    def _to_batch_documents(raw_text: Sequence[Sequence[str]] | Sequence[str]) -> list[list[str]]:
        if len(raw_text) == 0:
            logger.warning("LectureSegmentationModel received empty raw_text")
            raise ValueError("raw_text must not be empty")
        first_item = raw_text[0]
        if isinstance(first_item, str):
            return [list(raw_text)]
        return [list(document) for document in raw_text]

    def freeze_encoder(self) -> "LectureSegmentationModel":
        """Freeze the sentence encoder parameters and return ``self``."""

        self.encoder.requires_grad_(False)
        return self

    def freeze_transformer(self) -> "LectureSegmentationModel":
        """Freeze the transformer parameters and return ``self``."""

        self.transformer.requires_grad_(False)
        return self

    def freeze_detector(self) -> "LectureSegmentationModel":
        """Freeze the boundary detector parameters and return ``self``."""

        self.detector.requires_grad_(False)
        return self

    def forward(
        self,
        *,
        raw_text: Sequence[Sequence[str]] | Sequence[str] | None = None,
        sentence_embeddings: Tensor | None = None,
        transformer_features: Tensor | None = None,
        lengths: Tensor | list[int] | None = None,
        return_features: bool = False,
    ) -> Tensor | tuple[Tensor, dict[str, Tensor | list[int] | None]]:
        """Run one of the supported segmentation paths.

        Path A: raw text -> encoder -> transformer -> detector
        Path B: sentence_embeddings -> transformer -> detector
        Path C: transformer_features -> detector
        """

        cache: dict[str, Tensor | list[int] | None] = {}

        if transformer_features is not None:
            logger.debug("LectureSegmentationModel.forward path=C (transformer_features -> detector)")
            logits = self.detector(transformer_features, lengths=lengths)
            cache["transformer_features"] = transformer_features
            cache["lengths"] = lengths
            return (logits, cache) if return_features else logits

        if sentence_embeddings is not None:
            logger.debug("LectureSegmentationModel.forward path=B (sentence_embeddings -> transformer -> detector)")
            contextual = self.transformer(sentence_embeddings, lengths=lengths)
            logits = self.detector(contextual, lengths=lengths)
            cache["sentence_embeddings"] = sentence_embeddings
            cache["transformer_features"] = contextual
            cache["lengths"] = lengths
            return (logits, cache) if return_features else logits

        if raw_text is None:
            logger.warning("LectureSegmentationModel.forward called without any valid input")
            raise ValueError("Provide raw_text, sentence_embeddings, or transformer_features")

        logger.debug("LectureSegmentationModel.forward path=A (raw_text -> encoder -> transformer -> detector)")
        documents = self._to_batch_documents(raw_text)
        sentence_embeddings, inferred_lengths = self.encoder(documents, return_lengths=True)
        active_lengths = lengths if lengths is not None else inferred_lengths
        contextual = self.transformer(sentence_embeddings, lengths=active_lengths)
        logits = self.detector(contextual, lengths=active_lengths)
        cache["sentence_embeddings"] = sentence_embeddings
        cache["transformer_features"] = contextual
        cache["lengths"] = active_lengths
        return (logits, cache) if return_features else logits

    def save_checkpoint(self, path: str | Path) -> Path:
        """Save the full segmentation stack as a single checkpoint.

        Example:
            model.save_checkpoint("lecture_segmentation.pt")
        """

        target_path = Path(path)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "class_path": f"{self.__class__.__module__}.{self.__class__.__qualname__}",
                "config": self.get_config(),
                "submodules": {
                    "encoder": self.encoder._checkpoint_payload(),
                    "transformer": self.transformer._checkpoint_payload(),
                    "detector": self.detector._checkpoint_payload(),
                },
            },
            target_path,
        )
        logger.info("Saved LectureSegmentationModel checkpoint to %s", target_path)
        return target_path

    @classmethod
    def load_checkpoint(
        cls,
        path: str | Path,
        *,
        map_location: str | torch.device | None = "cpu",
    ) -> "LectureSegmentationModel":
        """Load a composite segmentation checkpoint."""

        source_path = Path(path)
        logger.info("Loading LectureSegmentationModel checkpoint from %s", source_path)
        payload = torch.load(source_path, map_location=map_location)
        if not isinstance(payload, dict) or "submodules" not in payload:
            raise TypeError(f"Unsupported segmentation checkpoint payload in {path}")

        encoder_payload = payload["submodules"]["encoder"]
        transformer_payload = payload["submodules"]["transformer"]
        detector_payload = payload["submodules"]["detector"]

        encoder = BaseModule._resolve_class(encoder_payload["class_path"]).from_config(encoder_payload["config"])
        transformer = BaseModule._resolve_class(transformer_payload["class_path"]).from_config(
            transformer_payload["config"]
        )
        detector = BaseModule._resolve_class(detector_payload["class_path"]).from_config(detector_payload["config"])

        encoder.load_state_dict(encoder_payload["state_dict"])
        transformer.load_state_dict(transformer_payload["state_dict"])
        detector.load_state_dict(detector_payload["state_dict"])
        logger.info("Loaded LectureSegmentationModel submodules from checkpoint")
        return cls(encoder=encoder, transformer=transformer, detector=detector)