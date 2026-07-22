from __future__ import annotations

from typing import TYPE_CHECKING, Sequence
from kpi.datasets.base_dataset import Dataset

if TYPE_CHECKING:
    from kpi.metrics import Metric
    from kpi.models.base_model import Model


def _canonicalize_boundaries(boundaries: list[float], duration: float) -> list[float]:
    interior = sorted({float(x) for x in boundaries if 0.0 < float(x) < float(duration)})
    return [0.0, *interior, float(duration)]


class Experiment:
    def __init__(
        self,
        test_data: Dataset,
        models: Sequence[Model],
        metrics: Sequence[Metric],
        plot_params: bool = True,
    ):
        self.test_data = test_data
        self.models = models
        self.metrics = metrics
        self.plot_params = plot_params

    def run(self) -> dict[tuple[str, str], float]:
        self.ret = {}
        self.preds = {}
        # print metric header
        print(f"{' '.join([str(metric) for metric in self.metrics])}")
        gt_with_caps = [
            _canonicalize_boundaries(list(frags), float(video.duration))
            for video, frags in zip(self.test_data.videos, self.test_data.frags)
        ]
        for model in self.models:
            cur_pred = model(self.test_data.videos)
            self.preds[str(model)] = cur_pred
            cur_ret = []
            pred_with_caps = [
                _canonicalize_boundaries(list(pred), float(video.duration))
                for pred, video in zip(cur_pred, self.test_data.videos)
            ]
            for metric in self.metrics:
                score = metric(
                    [pred[:] for pred in pred_with_caps],
                    [gt[:] for gt in gt_with_caps],
                )
                self.ret[(str(model), str(metric))] = score
                cur_ret.append(score)
            print(f"{str(model)} {' '.join([f'{x:.4f}' for x in cur_ret])}")

        return self.ret
