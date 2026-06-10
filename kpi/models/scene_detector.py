import click
from kpi.models.base_model import Model
from kpi.utils.video import Video
from scenedetect import detect, AdaptiveDetector, ContentDetector
from kpi.cli import register_model_runner


def run_predict_content(video_fn, threshold=27):
    detector = ContentDetector(threshold=threshold)
    ret = detect(video_fn, detector, show_progress=False)
    return [x[0].get_seconds() for x in ret]


def run_predict_adapt(video_fn, adaptive_threshold=3):
    detector = AdaptiveDetector(adaptive_threshold=adaptive_threshold)
    ret = detect(video_fn, detector, show_progress=False)
    return [x[0].get_seconds() for x in ret]


def search_predict(video_fn, k_frag, func):
    l = 0
    r = 256
    cur = []
    while r - l > 1:
        m = (l + r) // 2
        cur = func(video_fn, m)
        if len(cur) == k_frag:
            break
        elif len(cur) < k_frag:
            r = m
        else:
            l = m
    return cur


class SceneDetector(Model):
    CONTENT = "content"
    ADAPT = "adaptive"

    def __init__(self, detector, k):
        self.detector = detector
        self.k = k
        super().__init__()

    @property
    def model_name(self):
        return "PSD"

    @property
    def requires_training(self):
        return False

    @property
    def _if_print_progress(self):
        return True

    def fit(self, train_data, val_data=None):
        pass

    def _predict_one(self, video: Video):
        if self.detector == self.CONTENT:
            func = run_predict_content
        elif self.detector == self.ADAPT:
            func = run_predict_adapt
        else:
            raise ValueError("Unknown detector")
        ret = search_predict(video.video_fn, self.k, func)
        return sorted(ret)


register_model_runner(
    command_name="scd",
    model_cls=SceneDetector,
    model_param_build=lambda k, detector: dict(k=k, detector=detector),
    additional_options=[
        click.option("--k", type=int, required=True),
        click.option(
            "--detector",
            required=True,
            type=click.Choice([SceneDetector.CONTENT, SceneDetector.ADAPT]),
        ),
    ],
)
