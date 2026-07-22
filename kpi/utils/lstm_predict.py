import random
from pathlib import Path

import torch

from kpi.models.lstm import BiLSTM


def sec_to_timestamp(value: float) -> str:
    # Convert second-based timestamps into HH:MM:SS.mmm for readable output.
    total_ms = int(round(float(value) * 1000))
    hours = total_ms // 3_600_000
    minutes = (total_ms % 3_600_000) // 60_000
    seconds = (total_ms % 60_000) // 1000
    milliseconds = total_ms % 1000
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"


def format_fragments(frags: list[float | int]) -> str:
    # Render a full fragment boundary list as a comma-separated timestamp line.
    return ", ".join(sec_to_timestamp(float(x)) for x in frags)


def load_video_ids(dataset_path: str) -> list[str]:
    # Read canonical MITFLD video ids from video_id_list.txt in dataset root.
    id_file = Path(dataset_path) / "video_id_list.txt"
    with open(id_file, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def parse_requested_tokens(tokens: tuple[str, ...]) -> list[str]:
    # Support both repeated --video and comma-separated values in each token.
    parsed: list[str] = []
    for raw in tokens:
        for part in raw.split(","):
            cur = part.strip()
            if cur:
                parsed.append(cur)
    return parsed


def resolve_indices(
    requested_tokens: list[str],
    all_video_ids: list[str],
    num_videos: int,
    seed: int,
) -> list[int]:
    # Resolve selectors (id or index) and backfill the rest with seeded random picks.
    id_to_index = {vid: idx for idx, vid in enumerate(all_video_ids)}
    selected: list[int] = []
    used: set[int] = set()

    for token in requested_tokens:
        idx = None
        if token in id_to_index:
            idx = id_to_index[token]
        elif token.isdigit() and int(token) < len(all_video_ids):
            idx = int(token)
        else:
            raise ValueError(
                f"Unknown video token '{token}'. Provide a valid video ID or index."
            )

        if idx not in used:
            selected.append(idx)
            used.add(idx)

    if len(selected) < num_videos:
        rng = random.Random(seed)
        remaining = [i for i in range(len(all_video_ids)) if i not in used]
        rng.shuffle(remaining)
        need = num_videos - len(selected)
        selected.extend(remaining[:need])

    if len(selected) > num_videos:
        selected = selected[:num_videos]

    return selected


def build_model_from_weights(
    weights_path: str,
    visual_dim: int,
    audio_dim: int,
    text_dim: int,
    hidden_size: int,
    feat_func: str,
    batch_size: int,
) -> BiLSTM:
    # Load checkpoint metadata first to auto-align hidden_size/feature_keys when present.
    checkpoint = torch.load(weights_path, map_location="cpu")
    if isinstance(checkpoint, dict):
        meta = checkpoint.get("meta", {})
        hidden_size = int(meta.get("hidden_size", hidden_size))
        feat_func = str(meta.get("feature_keys", feat_func))

    model = BiLSTM(
        visual_dim=visual_dim,
        audio_dim=audio_dim,
        text_dim=text_dim,
        hidden_size=hidden_size,
        batch_size=batch_size,
        feat_func=feat_func,
        load_model_weights_path=weights_path,
    )
    # fit() is used here to trigger the existing load-and-skip-training behavior.
    model.fit(train_data=None)
    return model