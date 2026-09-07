"""Validate all six Phase 2B-2 score CSVs without recomputing experiments."""

from __future__ import annotations

import csv
import math
from pathlib import Path


SEED = 123456789
EXPECTED_ROWS = 131
EXPECTED_POSITIVE = 101
EXPECTED_NEGATIVE = 30
OUTPUT_DIR = Path("generated/results/phase2b2_sacbench_cifar10_s123456789")
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
RUNS = [
    ("AKH", "akh", 20),
    ("Random", "random", 20),
    ("IPGuard", "ipguard", 20),
    ("AKH", "akh", 50),
    ("Random", "random", 50),
    ("IPGuard", "ipguard", 50),
]


def parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    raise ValueError(f"invalid is_positive value: {value!r}")


def main() -> None:
    reference_pairs: set[tuple[str, str]] | None = None
    reference_metadata: dict[tuple[str, str], tuple[bool, str]] | None = None
    summaries: list[tuple[str, int, int, int]] = []

    for method, slug, budget in RUNS:
        path = OUTPUT_DIR / (
            f"phase2b2_sacbench_cifar10_{slug}_b{budget}_s{SEED}.csv"
        )
        if not path.is_file():
            raise FileNotFoundError(f"missing result file: {path}")

        with path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != EXPECTED_COLUMNS:
                raise ValueError(
                    f"{path}: unexpected columns {reader.fieldnames}; "
                    f"expected {EXPECTED_COLUMNS}"
                )
            rows = list(reader)

        if len(rows) != EXPECTED_ROWS:
            raise ValueError(f"{path}: expected {EXPECTED_ROWS} rows, found {len(rows)}")

        pairs: set[tuple[str, str]] = set()
        metadata: dict[tuple[str, str], tuple[bool, str]] = {}
        positive = 0
        for row in rows:
            if row["method"] != method:
                raise ValueError(f"{path}: unexpected method {row['method']!r}")
            if int(row["budget"]) != budget or int(row["seed"]) != SEED:
                raise ValueError(f"{path}: mixed budget or seed")
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

        if reference_pairs is None:
            reference_pairs = pairs
            reference_metadata = metadata
        elif pairs != reference_pairs or metadata != reference_metadata:
            raise ValueError(f"{path}: model pairs or labels differ from the first run")

        summaries.append((method, budget, positive, negative))

    print("method\tbudget\trows\tpositive\tnegative\tstatus")
    for method, budget, positive, negative in summaries:
        print(f"{method}\t{budget}\t{EXPECTED_ROWS}\t{positive}\t{negative}\tOK")
    print("All six Phase 2B-2 files use the same 131 model pairs and metadata.")


if __name__ == "__main__":
    main()

