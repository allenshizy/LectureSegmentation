import click
from typing import Iterable
from itertools import chain
from kpi.models.base_model import Model
from kpi.models.scene_detector import SceneDetector
from kpi.models.text_tiling import TextTiling
from kpi.models.voice_activity_detector import VoiceActivityDetector
from kpi.utils.video import Video
from kpi.cli import register_model_runner


def _TBM_predict_one(video: Video, models: Iterable[Model], threshold):
    preds = [model._predict_one(video) for model in models]
    all_points = list(chain(*preds))
    all_points.sort()
    ret = []
    for x in all_points:
        if not ret or x - ret[-1] > threshold:
            ret.append(x)
        else:
            last = ret.pop()
            ret.append((last + x) / 2)
    return sorted([i for i in ret if i < video.duration])


class TBM(Model):
    """Threshold-based Boundary Merge model"""

    def __init__(self, models: Iterable[Model], threshold=10):
        self.models = models
        self.threshold = threshold
        super().__init__()

    @property
    def requires_training(self):
        return True

    @property
    def _if_print_progress(self):
        return True

    def fit(self, train_data, val_data=None):
        pass

    def _predict_one(self, video: Video):
        return _TBM_predict_one(video, self.models, self.threshold)


def _get_models(kwargs: dict) -> list[Model]:
    models: list[Model] = []
    if kwargs["text_tiling"]:
        model_params = dict(
            value=kwargs["k"],
            policy=kwargs["policy"],
            window_type=kwargs["win_type"],
            window_size=kwargs["win_size"],
            window_step=kwargs["win_step"],
            feature_type=kwargs["feat_type"],
        )
        models.append(TextTiling(**model_params))
    if kwargs["scd"]:
        model_params = dict(k=kwargs["k"], detector=kwargs["detector"])
        models.append(SceneDetector(**model_params))
    if kwargs["vad"]:
        model_params = dict(k=kwargs["k"])
        models.append(VoiceActivityDetector(**model_params))
    return models


register_model_runner(
    command_name="tbm",
    model_cls=TBM,
    model_param_build=lambda kwargs: dict(models=_get_models(kwargs)),
    additional_options=[
        click.option("--k", type=int, required=True),
        click.option("--text_tiling", is_flag=True, default=False),
        click.option(
            "--policy", default="top_k", type=click.Choice(["threshold", "top_k"])
        ),
        click.option("--win_type", default="text", type=click.Choice(["srt", "text"])),
        click.option("--win_size", required=True, type=int),
        click.option("--win_step", required=True, type=int),
        click.option(
            "--feat_type", required=True, type=click.Choice(["word_embed", "sen_embed"])
        ),
        click.option("--scd", is_flag=True, default=False),
        click.option(
            "--detector", required=True, type=click.Choice(["content", "adaptive"])
        ),
        click.option("--vad", is_flag=True, default=False),
    ],
)
