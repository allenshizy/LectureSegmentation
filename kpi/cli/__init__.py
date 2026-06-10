import click

from kpi.datasets.avlecture import AVLecture
from kpi.datasets.mitfld import MITFLD
from kpi.experiments import Experiment
from kpi.metrics.f1 import F1
from kpi.metrics.iou import IoU
from kpi.metrics.mof import MoF
from kpi.utils import set_seed


def get_dataset(dataset_name: str, path: str):
    if dataset_name == "mitfld":
        return MITFLD(path)
    elif dataset_name == "avlecture":
        return AVLecture(path)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def global_options(func):
    @click.option(
        "--dataset", required=True, type=click.Choice(["mitfld", "avlecture"])
    )
    @click.option("--dataset_path", required=True, type=click.Path(exists=True))
    @click.option("--seed", default=2024)
    @click.option("--f1", default=True)
    @click.option("--f1_threshold", default=30)
    @click.option("--mof", default=False)
    @click.option("--mof_fps", default=10)
    @click.option("--iou", default=False)
    @click.option("--iou_fps", default=10)
    @click.option("--mmof", default=True)
    @click.option("--miou", default=True)
    def wrapped(*args, **kwargs):
        return func(*args, **kwargs)

    return wrapped


def single_model_experiment(*, model_cls, model_params, **kwargs):
    print(f"Running {model_cls.__name__} on {kwargs['dataset']}")

    set_seed(kwargs["seed"])

    dataset = get_dataset(kwargs["dataset"], kwargs["dataset_path"])
    train_data, val_data, test_data = dataset.random_split(
        [0.5, 0.25, 0.25],
        seed=kwargs["seed"],
    )
    model = model_cls(**model_params)
    if not model.requires_training:
        model.fit(train_data, val_data)

    metrics = []
    if kwargs["f1"]:
        metrics.append(F1(kwargs["f1_threshold"]))
    if kwargs["mof"]:
        metrics.append(MoF(kwargs["mof_fps"]))
    if kwargs["iou"]:
        metrics.append(IoU(kwargs["iou_fps"]))
    if kwargs["mmof"]:
        metrics.append(MoF(matching=True, fps=kwargs["mof_fps"]))
    if kwargs["miou"]:
        metrics.append(IoU(matching=True, fps=kwargs["iou_fps"]))

    method_exp = Experiment(
        test_data=test_data,
        models=[model],
        metrics=metrics,
    )
    method_exp.run()


@click.group()
def cli():
    pass


@click.group("model")
def run_model():
    pass


cli.add_command(run_model)


def register_model_runner(
    command_name, model_cls, model_param_build, additional_options=None
):
    """
    Registers a click command to run a single model experiment cleanly.

    Parameters:
        command_name: str, the CLI command name
        model_cls: your model class
        model_param_build: function(kwargs) -> dict for building model_params
        additional_options: list of click.option decorators if needed
    """

    def apply_options(f):
        if additional_options:
            for opt in reversed(additional_options):
                f = opt(f)
        return f

    @click.command(command_name)
    @apply_options
    @global_options
    def _runner(**kwargs):
        model_params = model_param_build(kwargs)
        single_model_experiment(
            model_cls=model_cls, model_params=model_params, **kwargs
        )

    run_model.add_command(_runner)
