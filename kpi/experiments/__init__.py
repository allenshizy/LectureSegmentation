from typing import Sequence
from kpi.datasets.base_dataset import Dataset
from kpi.models.base_model import Model
from kpi.metrics import Metric


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
        for model in self.models:
            cur_pred = model(self.test_data.videos)
            self.preds[str(model)] = cur_pred
            cur_ret = []
            for metric in self.metrics:
                score = metric(cur_pred, self.test_data.frags)
                self.ret[(str(model), str(metric))] = score
                cur_ret.append(score)
            print(f"{str(model)} {' '.join([f'{x:.4f}' for x in cur_ret])}")

        return self.ret
