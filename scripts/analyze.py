from pathlib import Path


import polars as pl
import plotly.express as px
from typer import Typer

from qurd.benchmark.analysis import roc_curve, roc_metrics

app = Typer(
    help="Analyze results obtained with the bench command",
    no_args_is_help=True,
    add_completion=False,
)
GENERATED_DIR = Path("generated/")


@app.command()
def main():
    scores = pl.read_csv(GENERATED_DIR / "scores.csv").with_columns(
        true_value=pl.col("source_model") == pl.col("target_model")
    )
    metrics = roc_metrics(
        scores,
        ["dataset", "method", "budget", "source_model"],
        score=1 - pl.col("score"),
        true_value=pl.col("source_model") == pl.col("target_model"),
        fpr_threshold=0.05,
    )
    print(metrics)

    roc = roc_curve(
        scores,
        ["dataset", "method", "budget", "source_model"],
        score=1 - pl.col("score"),
        true_value=pl.col("source_model") == pl.col("target_model"),
    )

    print(roc)
    fig = px.histogram(scores, x="score", facet_row="method", color="true_value")
    fig.show()
