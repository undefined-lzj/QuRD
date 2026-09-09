"""Validate Phase 2B-3 CSVs without running or recomputing experiments."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


EXPECTED_ROWS = 131
EXPECTED_POSITIVE = 101
EXPECTED_NEGATIVE = 30
RESULTS_ROOT = Path("generated/results/phase2b3_sacbench_cifar10")
BASELINE = Path(
    "generated/results/phase2/"
    "phase2_sacbench_cifar10_akh_b10_s123456789.csv"
)
EXPECTED_COLUMNS = [
    "method",
    "budget",
    "seed",
    "source_model",
    "target_model",
    "is_positive",
    "attack_type",
    "score",
    "dataset",
]
METHODS = [("AKH", "akh"), ("Random", "random"), ("IPGuard", "ipguard")]
BUDGETS = [10, 20, 50]


def parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    raise ValueError(f"invalid is_positive value: {value!r}")


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"missing result file: {path}")
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != EXPECTED_COLUMNS:
            raise ValueError(
                f"{path}: unexpected columns {reader.fieldnames}; "
                f"expected {EXPECTED_COLUMNS}"
            )
        return list(reader)


def validate_rows(
    path: Path,
    rows: list[dict[str, str]],
    *,
    method: str | None = None,
    budget: int | None = None,
    seed: int | None = None,
) -> tuple[set[tuple[str, str]], dict[tuple[str, str], tuple[bool, str]]]:
    if len(rows) != EXPECTED_ROWS:
        raise ValueError(f"{path}: expected {EXPECTED_ROWS} rows, found {len(rows)}")

    pairs: set[tuple[str, str]] = set()
    metadata: dict[tuple[str, str], tuple[bool, str]] = {}
    positive = 0

    for row in rows:
        if method is not None and row["method"] != method:
            raise ValueError(f"{path}: unexpected method {row['method']!r}")
        if budget is not None and int(row["budget"]) != budget:
            raise ValueError(f"{path}: unexpected or mixed budget")
        if seed is not None and int(row["seed"]) != seed:
            raise ValueError(f"{path}: unexpected or mixed seed")
        if row["dataset"] != "CIFAR10":
            raise ValueError(f"{path}: unexpected dataset {row['dataset']!r}")

        pair = (row["source_model"], row["target_model"])
        if pair in pairs:
            raise ValueError(f"{path}: duplicate model pair {pair}")
        pairs.add(pair)

        is_positive = parse_bool(row["is_positive"])
        positive += int(is_positive)
        metadata[pair] = (is_positive, row["attack_type"])

        score = float(row["score"])
        if not math.isfinite(score) or not 0.0 <= score <= 1.0:
            raise ValueError(f"{path}: invalid score {score!r} for {pair}")

    negative = len(rows) - positive
    if (positive, negative) != (EXPECTED_POSITIVE, EXPECTED_NEGATIVE):
        raise ValueError(
            f"{path}: expected {EXPECTED_POSITIVE}/{EXPECTED_NEGATIVE} "
            f"positive/negative pairs, found {positive}/{negative}"
        )
    return pairs, metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seed",
        action="append",
        type=int,
        choices=[42, 2026],
        required=True,
        help="Seed to validate; pass the option twice to validate both seeds together.",
    )
    args = parser.parse_args()
    seeds = list(dict.fromkeys(args.seed))

    baseline_rows = read_csv(BASELINE)
    baseline_pairs, baseline_metadata = validate_rows(BASELINE, baseline_rows)
    summaries: list[tuple[int, str, int]] = []

    for seed in seeds:
        for budget in BUDGETS:
            for method, slug in METHODS:
                path = RESULTS_ROOT / f"seed_{seed}" / (
                    f"phase2b3_sacbench_cifar10_{slug}_b{budget}_s{seed}.csv"
                )
                rows = read_csv(path)
                pairs, metadata = validate_rows(
                    path, rows, method=method, budget=budget, seed=seed
                )
                if pairs != baseline_pairs:
                    raise ValueError(f"{path}: model-pair set differs from the baseline")
                if metadata != baseline_metadata:
                    raise ValueError(
                        f"{path}: positive labels or attack types differ from the baseline"
                    )
                summaries.append((seed, method, budget))

    print("seed\tmethod\tbudget\trows\tpositive\tnegative\tstatus")
    for seed, method, budget in summaries:
        print(
            f"{seed}\t{method}\t{budget}\t{EXPECTED_ROWS}\t"
            f"{EXPECTED_POSITIVE}\t{EXPECTED_NEGATIVE}\tOK"
        )
    print(
        f"All {len(summaries)} selected files match the baseline 131 model pairs "
        "and metadata."
    )


if __name__ == "__main__":
    main()

