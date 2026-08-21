import logging
from logging import info, warning

from pathlib import Path
from typing import Annotated, TypedDict

import torch
import typer
import numpy as np
from dotenv import load_dotenv

from qurd.experiments import Experiment
from qurd.benchmark import get_benchmark
from qurd.fingerprint.fingerprints import make_fingerprint


class State(TypedDict):
    data_dir: Path
    models_dir: Path
    generated_dir: Path
    device: str
    batch_size: int
    seed: int


DEFAULT_DATA_DIR = Path("data/")
DEFAULT_MODELS_DIR = Path("generated/models/")
DEFAULT_GENERATED_DIR = Path("generated/")
DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

load_dotenv()
app = typer.Typer(
    pretty_exceptions_show_locals=False,
    help="Execute fingerprints on a benchmark",
    no_args_is_help=True,
    add_completion=False,
)
state = State(
    data_dir=DEFAULT_DATA_DIR,
    models_dir=DEFAULT_MODELS_DIR,
    generated_dir=DEFAULT_GENERATED_DIR,
    batch_size=128,
    device=DEFAULT_DEVICE,
    seed=123456789,
)


@app.command()
def list_models(benchmark: str):
    """List all the models (base models and variations) in a benchmark."""

    bench = get_benchmark(
        benchmark, state["data_dir"], state["models_dir"], state["device"]
    )

    for dataset in bench.datasets:
        for i, model_name in enumerate(bench.list_models(dataset)):
            print(f"{i+1}) {model_name}")
            _ = bench.torch_model(model_name)


@app.command()
def download(benchmark: str):
    """Download the benchmark models and datasets"""

    info("Preparing benchmark")
    bench = get_benchmark(
        benchmark, state["data_dir"], state["models_dir"], state["device"]
    )
    bench.prepare()


@app.command()
def eval_models(benchmark: str):
    """Print the accuracy of all the models and their variations"""

    bench = get_benchmark(
        benchmark, state["data_dir"], state["models_dir"], state["device"]
    )
    runner = Experiment(
        bench,
        dir=state["generated_dir"],
        batch_size=state["batch_size"],
        device=state["device"],
    )
    runner.eval_models()


@app.command()
def scores(benchmark: str, fingerprints: list[str], budget: int = 10):
    """
    Compute the fingerprint scores between all model pairs in a benchmark.

    For efficiency, the experiment runner handles the caches for the sampled
    queries and computed representations.
    """
    bench = get_benchmark(
        benchmark, state["data_dir"], state["models_dir"], state["device"]
    )
    runner = Experiment(
        bench,
        dir=state["generated_dir"],
        batch_size=state["batch_size"],
        device=state["device"],
    )
    runner.scores(
        {
            name: make_fingerprint(
                name, batch_size=state["batch_size"], device=state["device"]
            )
            for name in fingerprints
        },
        budget=budget,
    )
    info(f"Scores saved to {(state['generated_dir'] / 'scores.csv').resolve()}")


@app.callback()
def main(
    data_dir: Annotated[Path, typer.Option(envvar="DATA_DIR")] = DEFAULT_DATA_DIR,
    models_dir: Annotated[Path, typer.Option(envvar="MODELS_DIR")] = DEFAULT_MODELS_DIR,
    generated_dir: Annotated[
        Path, typer.Option(envvar="GENERATED_DIR")
    ] = DEFAULT_GENERATED_DIR,
    device: Annotated[str, typer.Option(envvar="DEVICE")] = DEFAULT_DEVICE,
    seed: int = 123456789,
    batch_size: int = 128,
    verbose: bool = False,
):

    torch.manual_seed(seed)
    np.random.seed(seed)
    state["seed"] = seed

    logging.basicConfig(
        format="%(asctime)s %(name)-12s %(levelname)-8s %(message)s",
        level=logging.DEBUG if verbose else logging.INFO,
    )
    logging.getLogger("timm").setLevel(logging.WARNING)

    if not data_dir.exists():
        warning(f"The provided data_dir {data_dir} does not exist, creating it.")
        data_dir.mkdir(parents=True, exist_ok=True)

    if not models_dir.exists():
        warning(
            f"The provided models_dir {models_dir.resolve()} does not exist, creating it."
        )
        models_dir.mkdir(parents=True, exist_ok=True)

    if not generated_dir.exists():
        warning(
            f"The provided generated_dir {generated_dir.resolve()} does not exist, creating it."
        )
        generated_dir.mkdir(parents=True, exist_ok=True)

    state["data_dir"] = data_dir
    state["models_dir"] = models_dir
    state["generated_dir"] = generated_dir

    state["batch_size"] = batch_size
    state["device"] = device

    info(f"{state = }")
