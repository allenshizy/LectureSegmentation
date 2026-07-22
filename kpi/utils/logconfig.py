import logging
from pathlib import Path

def setup_logging(level: str, path: str = None) -> Path:
    """Configure logging to console and file.

    The ``level`` argument supports two forms:
    - Single level name (for example: "INFO"): apply to root, console and file.
    - Three-letter profile (for example: "DID"): root, console, file respectively.
      Supported letters: D/I/W/E/C (DEBUG/INFO/WARNING/ERROR/CRITICAL).
    """

    level_map = {
        "D": logging.DEBUG,
        "I": logging.INFO,
        "W": logging.WARNING,
        "E": logging.ERROR,
        "C": logging.CRITICAL,
    }

    normalized = level.strip().upper()
    if len(normalized) == 3 and all(ch in level_map for ch in normalized):
        root_level, console_level, file_level = (level_map[ch] for ch in normalized)
    else:
        resolved_level = getattr(logging, normalized, None)
        if not isinstance(resolved_level, int):
            raise ValueError(
                "Invalid level value. Use a logging level name (e.g. INFO) "
                "or a 3-letter profile with D/I/W/E/C (e.g. DID)."
            )
        root_level = console_level = file_level = resolved_level

    repo_dir = Path(__file__).resolve().parent.parent.parent
    if path is not None:
        log_path = Path(path) / "train.log"
    else:
        log_path = repo_dir / "train.log"
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    root_logger = logging.getLogger()
    root_logger.setLevel(root_level)
    root_logger.handlers.clear()

    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)
    console_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    file_handler.setLevel(file_level)
    file_handler.setFormatter(formatter)

    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    root_logger.info("Logging initialized. Log file: %s", log_path)
    return log_path