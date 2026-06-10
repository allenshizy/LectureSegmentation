import numpy as np
import torch
import click

import kpi.models._twfinch as _twfinch
from kpi.models._jvte import JVTE
from kpi.models.base_model import Model
from kpi.utils.preprocessing import get_video_feature_clip
from kpi.utils.video import Video
from kpi.cli import register_model_runner


class TWFinchJVTE(Model):
    def __init__(
        self,
        dataset,
        k=None,
        last_partition=None,
        slice_method="srt",
        slice_duration=None,
        v_dim=None,
        t_dim=None,
        feat_dim=None,
        sigma=0.1,
        batch=32,
        lr=1e-3,
        epochs=100,
        hard_neg=1,
        soft_neg=1,
        device="cuda",
    ):
        if k:
            self.k = k
        elif last_partition == 2 or last_partition == 3:
            self.last_partition = last_partition
        else:
            raise ValueError("k or last_partition must be provided")
        self.slice_method = slice_method
        self.slice_duration = slice_duration
        self._jvte = JVTE(
            v_dim,
            t_dim,
            feat_dim,
            sigma,
            batch,
            lr,
            epochs,
            dataset=dataset,
            device=device,
            hard_neg=hard_neg,
            soft_neg=soft_neg,
        )
        self._jvte = self._jvte.to(device)

        self._trained = False

        super().__init__()

    @property
    def requires_training(self):
        return not self._trained

    def fit(self, train_data, val_data=None):
        self._jvte.fit(train_data, val_data)
        self._trained = True

    def _predict_one(self, video: Video):
        clips = get_video_feature_clip(
            video, self.slice_method, duration=self.slice_duration
        )
        with torch.no_grad():
            vectors = np.array([self._jvte(x).cpu() for x in clips])
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
    command_name="tw_finch_jvte",
    model_cls=TWFinchJVTE,
    model_param_build=lambda kwargs: dict(
        k=kwargs["k"],
        last_partition=kwargs["last_partition"],
        slice_method=kwargs["slice_method"],
        slice_duration=kwargs["slice_duration"],
        t_dim=kwargs["text_dim"],
        v_dim=kwargs["visual_dim"],
        feat_dim=kwargs["feat_dim"],
        sigma=0.1,
        batch=kwargs["batch_size"],
        lr=kwargs["lr"],
        epochs=kwargs["epochs"],
        hard_neg=kwargs["hard_neg"],
        soft_neg=kwargs["soft_neg"],
        dataset=kwargs["dataset"],
    ),
    additional_options=[
        click.option("--k", type=int),
        click.option("--last_partition", type=int),
        click.option(
            "--slice_method", type=click.Choice(["srt", "text"]), default="srt"
        ),
        click.option("--slice_duration", type=int, default=10),
        click.option("--visual_dim", type=int, default=1000),
        click.option("--audio_dim", type=int, default=512),
        click.option("--text_dim", type=int, default=512),
        click.option("--feat_dim", type=int, default=1024),
        click.option("--batch_size", type=int, default=32),
        click.option("--lr", type=float, default=1e-3),
        click.option("--epochs", type=int, default=100),
        click.option("--hard_neg", type=int, default=1),
        click.option("--soft_neg", type=int, default=1),
    ],
)
