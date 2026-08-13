import sys
from pathlib import Path

import click

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kpi.datasets.mitfld import MITFLD
from kpi.experiments import Experiment
from kpi.metrics.f1 import F1
from kpi.metrics.iou import IoU
from kpi.metrics.mof import MoF
from kpi.models.lstm import BiLSTM
from kpi.utils import set_seed


SEED = 2024


@click.command()
@click.option(
    "--dataset_path",
    required=True,
    type=str,
    help="Path to the dataset root directory.",
)
@click.option("--visual_dim", type=int, default=1000)
@click.option("--audio_dim", type=int, default=512)
@click.option("--text_dim", type=int, default=512)
@click.option("--hidden_size", type=int, default=256)
@click.option("--batch_size", type=int, default=32)
@click.option("--epochs", type=int, default=20)
@click.option("--lr", type=float, default=1e-3)
@click.option("--neg_weight", type=float, default=0.01)
@click.option(
    "--feat_func",
    type=click.Choice(["t", "tv", "ta", "tva", "v", "va", "a"]),
    default="tva",
)
@click.option(
    "--save_feature_vectors/--no-save_feature_vectors",
    default=False,
    help="Save train/val/test feature vectors used by this run.",
)
@click.option(
    "--load_feature_vectors_path",
    type=str,
    default="",
    help="Load feature vectors from a train split .npz path and infer val/test paths.",
)
@click.option(
    "--save_model_weights/--no-save_model_weights",
    default=False,
    help="Save trained BiLSTM weights after fit().",
)
@click.option(
    "--load_model_weights_path",
    type=str,
    default="",
    help="Load BiLSTM weights from file and skip training.",
)
@click.option(
    "--artifact_dir",
    type=str,
    default="./artifacts/lstm",
    help="Directory for saved features and weights.",
)
@click.option(
    "--artifact_prefix",
    type=str,
    default="demo_lstm",
    help="File prefix used in artifact naming.",
)
def run_demo_lstm_experiment(
    dataset_path,
    visual_dim,
    audio_dim,
    text_dim,
    hidden_size,
    batch_size,
    epochs,
    lr,
    neg_weight,
    feat_func,
    save_feature_vectors,
    load_feature_vectors_path,
    save_model_weights,
    load_model_weights_path,
    artifact_dir,
    artifact_prefix,
):
    print("Running demo experiment: BiLSTM on MITFLD")

    set_seed(SEED)

    dataset = MITFLD(dataset_path)
    train_data, val_data, test_data = dataset.random_split(
        [0.5, 0.25, 0.25],
        seed=SEED,
    )

    model = BiLSTM(
        visual_dim=visual_dim,
        audio_dim=audio_dim,
        text_dim=text_dim,
        hidden_size=hidden_size,
        batch_size=batch_size,
        epochs=epochs,
        lr=lr,
        feat_func=feat_func,
        weight=[neg_weight, 1 - neg_weight],
        save_feature_vectors=save_feature_vectors,
        load_feature_vectors_path=load_feature_vectors_path,
        save_model_weights=save_model_weights,
        load_model_weights_path=load_model_weights_path,
        artifact_dir=artifact_dir,
        artifact_prefix=artifact_prefix,
    )

    # BiLSTM is supervised and must be trained before evaluation.
    model.fit(train_data, val_data, test_data)

    metrics = [
        F1(threshold=30),
        MoF(matching=True, fps=10),
        IoU(matching=True, fps=10),
    ]

    method_exp = Experiment(
        test_data=test_data,
        models=[model],
        metrics=metrics,
    )
    method_exp.run()


if __name__ == "__main__":
    run_demo_lstm_experiment()
