from __future__ import annotations

import importlib
import json
from abc import ABC
from pathlib import Path
from typing import Any, Self
import logging

import torch
from torch import Tensor, nn


def _as_path(path: str | Path) -> Path:
    return path if isinstance(path, Path) else Path(path)


logger = logging.getLogger(__name__)

class BaseModule(nn.Module, ABC):
    """Base class for STB segmentation modules.

    Every concrete module inherits a small persistence surface:

    - forward()
    - save_weights() / load_weights()
    - save_config() / load_config()
    - save_checkpoint() / load_checkpoint()
    - save_features() / load_features()

    All config payloads are JSON-serializable and checkpoints store both config
    and state_dict so a module can be reconstructed automatically.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__()
        self.config: dict[str, Any] = dict(config or {})
        logger.debug("Initialized %s with config keys: %s", self.__class__.__name__, sorted(self.config.keys()))

    def get_config(self) -> dict[str, Any]:
        """Return a JSON-serializable configuration dictionary."""

        return dict(self.config)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> Self:
        """Construct a module from a configuration dictionary."""

        logger.info("Constructing %s from config", cls.__name__)
        return cls(**config)

    @staticmethod
    def _resolve_class(class_path: str) -> type[BaseModule]:
        logger.debug("Resolving class path: %s", class_path)
        module_name, class_name = class_path.rsplit(".", 1)
        module = importlib.import_module(module_name)
        target = getattr(module, class_name)
        if not issubclass(target, BaseModule):
            raise TypeError(f"{class_path} is not a BaseModule subclass")
        return target

    def _checkpoint_payload(self) -> dict[str, Any]:
        return {
            "class_path": f"{self.__class__.__module__}.{self.__class__.__qualname__}",
            "config": self.get_config(),
            "state_dict": self.state_dict(),
        }

    def save_config(self, path: str | Path) -> Path:
        """Save module configuration as JSON.

        Example:
            encoder.save_config("encoder_config.json")
        """

        target_path = _as_path(path)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        with target_path.open("w", encoding="utf-8") as handle:
            json.dump(self.get_config(), handle, indent=2, sort_keys=True)
        logger.info("Saved %s config to %s", self.__class__.__name__, target_path)
        return target_path

    @classmethod
    def load_config(cls, path: str | Path) -> dict[str, Any]:
        """Load a JSON config file or a checkpoint payload config.

        Example:
            config = SentenceEncoder.load_config("encoder_config.json")
        """

        target_path = _as_path(path)
        logger.info("Loading config for %s from %s", cls.__name__, target_path)
        if target_path.suffix.lower() == ".json":
            with target_path.open("r", encoding="utf-8") as handle:
                return json.load(handle)

        payload = torch.load(target_path, map_location="cpu")
        if isinstance(payload, dict) and "config" in payload:
            return dict(payload["config"])
        if isinstance(payload, dict):
            return dict(payload)
        raise TypeError(f"Unsupported config payload in {target_path}")

    def save_weights(self, path: str | Path) -> Path:
        """Save model weights together with config metadata.

        Example:
            detector.save_weights("detector_weights.pt")
        """

        target_path = _as_path(path)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self._checkpoint_payload(), target_path)
        logger.info("Saved %s weights to %s", self.__class__.__name__, target_path)
        return target_path

    def load_weights(
        self,
        path: str | Path,
        *,
        map_location: str | torch.device | None = "cpu",
        strict: bool = True,
    ) -> Self:
        """Load weights into the current instance.

        Example:
            detector.load_weights("detector_weights.pt")
        """

        source_path = _as_path(path)
        logger.info(
            "Loading weights into %s from %s (strict=%s)",
            self.__class__.__name__,
            source_path,
            strict,
        )
        payload = torch.load(source_path, map_location=map_location)
        if isinstance(payload, dict) and "state_dict" in payload:
            state_dict = payload["state_dict"]
        else:
            state_dict = payload
        self.load_state_dict(state_dict, strict=strict)
        return self

    def save_checkpoint(self, path: str | Path) -> Path:
        """Save a full reconstruction checkpoint.

        Example:
            encoder.save_checkpoint("encoder_checkpoint.pt")
        """

        return self.save_weights(path)

    @classmethod
    def from_checkpoint_payload(
        cls: type[Self],
        payload: dict[str, Any],
    ) -> Self:
        """Rebuild a module from an in-memory checkpoint payload."""

        target_cls: type[BaseModule]
        if cls is BaseModule:
            target_cls = cls._resolve_class(payload["class_path"])
        else:
            target_cls = cls
        module = target_cls.from_config(dict(payload["config"]))
        module.load_state_dict(payload["state_dict"])
        return module

    @classmethod
    def load_checkpoint(
        cls: type[Self],
        path: str | Path,
        *,
        map_location: str | torch.device | None = "cpu",
    ) -> Self:
        """Load a checkpoint and reconstruct the module automatically.

        Example:
            encoder = SentenceEncoder.load_checkpoint("encoder_checkpoint.pt")
        """

        source_path = _as_path(path)
        logger.info("Loading checkpoint for %s from %s", cls.__name__, source_path)
        payload = torch.load(source_path, map_location=map_location)
        if not isinstance(payload, dict) or "state_dict" not in payload:
            raise TypeError(f"Unsupported checkpoint payload in {path}")
        return cls.from_checkpoint_payload(payload)

    def save_features(
        self,
        features: Tensor,
        path: str | Path,
        *,
        lengths: Tensor | list[int] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Path:
        """Save cached features to a .pt file.

        Example:
            SentenceEncoder.save_features(embeddings, "embeddings.pt", lengths=lengths)
        """

        target_path = _as_path(path)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {
            "features": features.detach().cpu(),
            "metadata": metadata or {},
        }
        if lengths is not None:
            payload["lengths"] = torch.as_tensor(lengths, dtype=torch.long).cpu()
        torch.save(payload, target_path)
        logger.info(
            "Saved %s features to %s with shape=%s",
            self.__class__.__name__,
            target_path,
            tuple(features.shape),
        )
        return target_path

    @staticmethod
    def load_features(
        path: str | Path,
        *,
        map_location: str | torch.device | None = "cpu",
    ) -> dict[str, Any]:
        """Load cached features from a .pt file.

        Example:
            cached = BaseModule.load_features("embeddings.pt")
        """

        source_path = _as_path(path)
        logger.info("Loading features from %s", source_path)
        payload = torch.load(source_path, map_location=map_location)
        if isinstance(payload, Tensor):
            return {"features": payload, "lengths": None, "metadata": {}}
        if not isinstance(payload, dict) or "features" not in payload:
            raise TypeError(f"Unsupported feature payload in {path}")
        return {
            "features": payload["features"],
            "lengths": payload.get("lengths"),
            "metadata": payload.get("metadata", {}),
        }


def lengths_to_padding_mask(lengths: Tensor | list[int], max_len: int) -> Tensor:
    """Build a boolean padding mask from sequence lengths.

    Returns True at padded positions.
    """

    if not torch.is_tensor(lengths):
        lengths = torch.as_tensor(lengths, dtype=torch.long)
    positions = torch.arange(max_len, device=lengths.device).unsqueeze(0)
    return positions >= lengths.unsqueeze(1)

