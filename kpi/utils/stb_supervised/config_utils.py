"""Config and runtime helpers for supervised STB scripts."""

from __future__ import annotations

import json
import logging
from typing import Any

import click
import torch


logger = logging.getLogger(__name__)


def setup_logging(level: str) -> None:
    """Initialize root logging with a consistent format."""

    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )
    logger.info("STB supervised logging initialized at level=%s", level.upper())


def parse_json_kwargs(option_name: str, raw_value: str) -> dict[str, Any]:
    """Parse a click JSON option and ensure it decodes to a dict."""

    logger.debug("Parsing JSON kwargs for option=%s", option_name)
    try:
        parsed = json.loads(raw_value)
    except json.JSONDecodeError as exc:
        logger.error("Invalid JSON for option=%s: %s", option_name, exc)
        raise click.BadParameter(
            f"{option_name} must be a valid JSON object string. error={exc}"
        ) from exc

    if not isinstance(parsed, dict):
        logger.error("JSON option=%s did not decode to object type", option_name)
        raise click.BadParameter(f"{option_name} must decode to a JSON object")
    logger.debug("Parsed JSON kwargs for option=%s with %d keys", option_name, len(parsed))
    return parsed


def resolve_device(device: str) -> torch.device:
    """Resolve a user device flag into a torch.device."""

    if device == "auto":
        resolved = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Resolved device=auto to %s", resolved)
        return resolved

    resolved = torch.device(device)
    logger.info("Resolved explicit device to %s", resolved)
    return resolved


def validate_split_ratios(train_ratio: float, val_ratio: float, test_ratio: float) -> list[float]:
    """Validate positive split ratios that sum to 1.0."""

    ratios = [train_ratio, val_ratio, test_ratio]
    logger.debug(
        "Validating split ratios train=%.4f val=%.4f test=%.4f",
        train_ratio,
        val_ratio,
        test_ratio,
    )
    if any(r <= 0 for r in ratios):
        logger.error("Split ratio validation failed: non-positive value(s) %s", ratios)
        raise click.BadParameter("train_ratio, val_ratio, and test_ratio must all be > 0")
    ratio_sum = sum(ratios)
    if abs(ratio_sum - 1.0) > 1e-6:
        logger.error("Split ratio validation failed: sum=%.6f", ratio_sum)
        raise click.BadParameter(
            f"train_ratio + val_ratio + test_ratio must equal 1.0, got {ratio_sum:.6f}"
        )
    logger.info("Split ratios validated successfully")
    return ratios
