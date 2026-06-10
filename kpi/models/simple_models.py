import random
import click
from kpi.models.base_model import Model
from kpi.utils.video import Video
from kpi.cli import register_model_runner


class RandomModel(Model):
    def __init__(
        self,
        seed,
        num_frags: None | int = None,
        dur_frag: None | float = None,
    ):
        if dur_frag and num_frags or (not dur_frag and not num_frags):
            raise ValueError("Either dur_frag or num_frags should be provided")
        self.num_frags = num_frags
        self.dur_frag = dur_frag
        self.seed = seed

    @property
    def requires_training(self):
        return False

    def fit(self, train_data, val_data=None):
        pass

    def _predict_one(self, video: Video):
        duration = video.duration

        num_frags = self.num_frags
        dur_frag = self.dur_frag

        if dur_frag:
            num_frags = max(int(duration / dur_frag), 1)

        assert num_frags
        ret = [
            0,
            *(random.uniform(0 + 0.1, duration - 0.1) for _ in range(num_frags - 1)),
            duration,
        ]
        return sorted(ret)

    def predict(self, videos: list[Video]):
        random.seed(self.seed)
        return super().predict(videos)


class EvenlyModel(Model):
    def __init__(
        self,
        k: int,
    ):
        if not k:
            raise ValueError("k, the number of fragments, should be provided")
        self.num_frags = k

    @property
    def requires_training(self):
        return True

    @property
    def model_name(self):
        return "Evenly"

    def fit(self, train_data, val_data=None):
        pass

    def _predict_one(self, video: Video):
        duration = video.duration

        dur_frag = duration / self.num_frags

        return [
            0,
            *(i * dur_frag for i in range(1, int(duration / dur_frag))),
            duration,
        ]

    def predict(self, videos: list[Video]):
        return super().predict(videos)


register_model_runner(
    command_name="evenly",
    model_cls=EvenlyModel,
    model_param_build=lambda kwargs: dict(k=kwargs["k"]),
    additional_options=[
        click.option("--k", type=int, required=True, help="Number of fragments.")
    ],
)
