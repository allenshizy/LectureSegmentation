import numpy as np
import click

import kpi.models._twfinch as _twfinch
from kpi.models.base_model import Model
from kpi.utils.preprocessing import FEATURE_FUNC, get_video_feature_clip
from kpi.utils.video import Video
from kpi.cli import register_model_runner


class TWFinch(Model):
    def __init__(
        self,
        k=None,
        last_partition=None,
        slice_method="srt",
        slice_duration=None,
        feat_func=None,
    ):
        if k:
            self.k = k
        elif last_partition == 2 or last_partition == 3:
            self.last_partition = last_partition
        else:
            raise ValueError("k or last_partition must be provided")
        if not feat_func:
            raise ValueError("feat_func must be provided")
        self.slice_method = slice_method
        self.slice_duration = slice_duration
        self.feat_func = feat_func
        super().__init__()

    @property
    def requires_training(self):
        return False

    def fit(self, train_data, val_data=None):
        pass

    @property
    def _if_print_progress(self):
        return True

    def _predict_one(self, video: Video):
        clips = get_video_feature_clip(
            video, self.slice_method, duration=self.slice_duration
        )
        vectors = np.array([self.feat_func(x) for x in clips])
        if getattr(self, "k", None):
            c, num_clust, req_c = _twfinch.FINCH(
                vectors, req_clust=self.k, tw_finch=True
            )
            labels = req_c
        else:
            c, num_clust, req_c = _twfinch.FINCH(vectors, tw_finch=True)
            labels = [x[-1 * min(self.last_partition, len(x))] for x in c]
        ret = []
        last_label = -1
        for label, clip in zip(labels, clips):  # type: ignore
            if label != last_label:
                ret.append(clip.start)
                last_label = label
        return sorted([i for i in ret if i < video.duration])


register_model_runner(
    command_name="tw_finch",
    model_cls=TWFinch,
    model_param_build=lambda kwargs: dict(
        k=kwargs["k"],
        last_partition=kwargs["last_partition"],
        slice_method=kwargs["slice_method"],
        slice_duration=kwargs["slice_duration"],
        feat_func=FEATURE_FUNC[kwargs["feat_func"]],
    ),
    additional_options=[
        click.option("--k", type=int),
        click.option("--last_partition", type=int),
        click.option(
            "--slice_method", type=click.Choice(["srt", "text"]), default="srt"
        ),
        click.option("--slice_duration", type=int, default=10),
        click.option(
            "--feat_func", type=click.Choice(list(FEATURE_FUNC.keys())), required=True
        ),
    ],
)
