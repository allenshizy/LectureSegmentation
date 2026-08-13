import os
import click

from kpi.datasets.mitfld import MITFLD
from kpi.metrics.f1 import F1
from kpi.metrics.iou import IoU
from kpi.metrics.mof import MoF
from kpi.utils import set_seed
from kpi.models.simple_models import EvenlyModel
from kpi.models.text_tiling import TextTiling
from kpi.models.voice_activity_detector import VoiceActivityDetector
from kpi.experiments import Experiment


SEED = 2024


@click.command()
@click.option(
    "--dataset_path",
    required=True,
    type=str,
    help="Path to the dataset root directory.",
)
def run_demo_expriment(dataset_path):
    print("Running demo experiment: Evenly, SceneDection on MITLFD")

    set_seed(SEED)

    dataset = MITFLD(dataset_path)
    train_data, val_data, test_data = dataset.random_split(
        [0.5, 0.25, 0.25],
        seed=SEED,
    )

    # Average number of fragments per lecture for mitfld: 13.84
    models = [
        EvenlyModel(k=14),
        VoiceActivityDetector(k=14),
        TextTiling(
            value=14,
            policy=TextTiling.POLICY_TOP_K,
            window_size=120,
            window_step=20,
            window_type=TextTiling.WINDOW_TYPE_TEXT,
            feature_type=TextTiling.FEATURE_TYPE_SEN,
        ),
    ]

    metrics = [
        F1(threshold=30),
        MoF(matching=True, fps=10),
        IoU(matching=True, fps=10),
    ]

    method_exp = Experiment(
        test_data=test_data,
        models=models,
        metrics=metrics,
    )
    method_exp.run()


if __name__ == "__main__":
    run_demo_expriment()
