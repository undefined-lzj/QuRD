from pathlib import Path
from typing import Annotated


import polars as pl
import plotly.express as px
from typer import Option, Typer

from qurd.benchmark.analysis import roc_curve, roc_metrics

app = Typer(
    help="Analyze results obtained with the bench command",
    no_args_is_help=True,
    add_completion=False,
)
GENERATED_DIR = Path("generated/")


@app.command()
def main(
    input_file: Annotated[
        Path,
        Option("--input", "-i", help="CSV file produced by a benchmark run"),
    ] = GENERATED_DIR / "scores.csv",
):
    scores = pl.read_csv(input_file)
    metrics = roc_metrics(
        scores,
        ["dataset", "method", "budget", "seed", "source_model"],
        score=1 - pl.col("score"),
        true_value=pl.col("is_positive"),
        fpr_threshold=0.05,
    )
    print(metrics)

    roc = roc_curve(
        scores,
        ["dataset", "method", "budget", "seed", "source_model"],
        score=1 - pl.col("score"),
        true_value=pl.col("is_positive"),
    )

    print(roc)
    fig = px.histogram(scores, x="score", facet_row="method", color="true_value")
    fig.show()
