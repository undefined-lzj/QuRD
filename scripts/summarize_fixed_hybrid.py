"""Validate and summarize the Phase 3 Fixed-Hybrid result CSV."""

from __future__ import annotations

import csv
import math
from pathlib import Path

INPUT = Path("generated/results/phase3_fixed_hybrid_cifar10_b20_s123456789.csv")
OUTPUT = Path(
    "generated/results/phase3_fixed_hybrid_cifar10_b20_s123456789_summary.csv"
)
AKH_BASELINE = Path(
    "generated/results/phase2b2_sacbench_cifar10_s123456789/"
    "phase2b2_sacbench_cifar10_akh_b20_s123456789.csv"
)
IPGUARD_BASELINE = Path(
    "generated/results/phase2b2_sacbench_cifar10_s123456789/"
    "phase2b2_sacbench_cifar10_ipguard_b20_s123456789.csv"
)
EXPECTED_SPLITS = [(0, 20), (5, 15), (10, 10), (15, 5), (20, 0)]
FIXED_HYBRID_COLUMNS = [
    "method",
    "budget",
    "seed",
    "source_model",
    "target_model",
    "score",
    "dataset",
    "pair_label",
    "attack_type",
    "akh_budget",
    "ipguard_budget",
    "akh_ratio",
    "akh_score",
    "ipguard_score",
    "hybrid_score",
]
SUMMARY_COLUMNS = [
    "akh_budget",
    "ipguard_budget",
    "akh_ratio",
    "n_pairs",
    "auc",
    "tpr_at_5pct_fpr",
    "achieved_fpr",
    "positive_mean_score",
    "negative_mean_score",
    "score_gap",
]


def read_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def pair_key(row: dict[str, str]) -> tuple[str, str]:
    return row["source_model"], row["target_model"]


def auc_with_ties(rows: list[dict[str, str]]) -> float:
    ranked = sorted(
        ((int(row["pair_label"]), -float(row["score"])) for row in rows),
        key=lambda item: item[1],
    )
    positive_rank_sum = 0.0
    start = 0
    while start < len(ranked):
        end = start + 1
        while end < len(ranked) and ranked[end][1] == ranked[start][1]:
            end += 1
        average_rank = ((start + 1) + end) / 2
        positive_rank_sum += average_rank * sum(
            ranked[index][0] for index in range(start, end)
        )
        start = end
    n_positive = sum(label for label, _ in ranked)
    n_negative = len(ranked) - n_positive
    return (
        positive_rank_sum - n_positive * (n_positive + 1) / 2
    ) / (n_positive * n_negative)


def operating_point(
    rows: list[dict[str, str]], fpr_limit: float = 0.05
) -> tuple[float, float]:
    n_positive = sum(int(row["pair_label"]) for row in rows)
    n_negative = len(rows) - n_positive
    thresholds = [math.inf] + sorted(
        {-float(row["score"]) for row in rows}, reverse=True
    )
    best_tpr = 0.0
    best_fpr = 0.0
    for threshold in thresholds:
        true_positive = false_positive = 0
        for row in rows:
            predicted_positive = -float(row["score"]) >= threshold
            label = int(row["pair_label"])
            true_positive += int(predicted_positive and label == 1)
            false_positive += int(predicted_positive and label == 0)
        tpr = true_positive / n_positive
        fpr = false_positive / n_negative
        if fpr <= fpr_limit and tpr > best_tpr:
            best_tpr = tpr
            best_fpr = fpr
    return best_tpr, best_fpr


def baseline_index(path: Path) -> dict[tuple[str, str], dict[str, str]]:
    columns, rows = read_rows(path)
    required = {
        "source_model",
        "target_model",
        "is_positive",
        "attack_type",
        "score",
    }
    if not required.issubset(columns) or len(rows) != 131:
        raise ValueError(f"Invalid baseline CSV: {path}")
    index = {pair_key(row): row for row in rows}
    if len(index) != 131:
        raise ValueError(f"Duplicate model pairs in baseline CSV: {path}")
    return index


def main() -> None:
    columns, rows = read_rows(INPUT)
    if columns != FIXED_HYBRID_COLUMNS:
        raise ValueError(f"Unexpected Fixed-Hybrid columns: {columns}")
    if len(rows) != 655:
        raise ValueError(f"Expected 655 rows, found {len(rows)}")

    groups: dict[tuple[int, int], list[dict[str, str]]] = {
        split: [] for split in EXPECTED_SPLITS
    }
    reference_pairs: set[tuple[str, str]] | None = None
    reference_metadata: dict[tuple[str, str], tuple[int, str]] | None = None

    for row in rows:
        if (
            row["method"] != "Fixed-Hybrid"
            or int(row["budget"]) != 20
            or int(row["seed"]) != 123456789
            or row["dataset"] != "CIFAR10"
        ):
            raise ValueError("Fixed-Hybrid CSV mixes experiment configurations")
        split = (int(row["akh_budget"]), int(row["ipguard_budget"]))
        if split not in groups:
            raise ValueError(f"Unexpected component budgets: {split}")
        if sum(split) != 20 or abs(float(row["akh_ratio"]) - split[0] / 20) > 1e-12:
            raise ValueError(f"Invalid budget or ratio in row: {row}")

        score = float(row["score"])
        hybrid_score = float(row["hybrid_score"])
        if not 0 <= score <= 1 or abs(score - hybrid_score) > 1e-12:
            raise ValueError("score must be a valid distance equal to hybrid_score")
        akh_score = None if row["akh_score"] == "" else float(row["akh_score"])
        ipguard_score = (
            None if row["ipguard_score"] == "" else float(row["ipguard_score"])
        )
        if split[0] == 0 and akh_score is not None:
            raise ValueError("akh_score must be blank when akh_budget=0")
        if split[1] == 0 and ipguard_score is not None:
            raise ValueError("ipguard_score must be blank when ipguard_budget=0")
        if split[0] and split[1]:
            if akh_score is None or ipguard_score is None:
                raise ValueError("Both component scores are required for a mixed split")
            expected = (akh_score * split[0] + ipguard_score * split[1]) / 20
            if abs(score - expected) > 1e-12:
                raise ValueError("hybrid_score is not the weighted component score")
        elif split[0]:
            if akh_score is None or abs(score - akh_score) > 1e-12:
                raise ValueError("AKH-only hybrid score differs from akh_score")
        elif split[1]:
            if ipguard_score is None or abs(score - ipguard_score) > 1e-12:
                raise ValueError(
                    "IPGuard-only hybrid score differs from ipguard_score"
                )
        groups[split].append(row)

    for split, group in groups.items():
        if len(group) != 131:
            raise ValueError(f"Split {split} has {len(group)} rows instead of 131")
        pairs = {pair_key(row) for row in group}
        metadata = {
            pair_key(row): (int(row["pair_label"]), row["attack_type"])
            for row in group
        }
        if len(pairs) != 131 or len(metadata) != 131:
            raise ValueError(f"Split {split} contains duplicate model pairs")
        if sum(int(row["pair_label"]) for row in group) != 101:
            raise ValueError(f"Split {split} does not contain 101 positive pairs")
        if reference_pairs is None:
            reference_pairs = pairs
            reference_metadata = metadata
        elif pairs != reference_pairs or metadata != reference_metadata:
            raise ValueError(f"Split {split} changes model pairs or metadata")

    akh_baseline = baseline_index(AKH_BASELINE)
    ipguard_baseline = baseline_index(IPGUARD_BASELINE)
    for split, baseline in [((20, 0), akh_baseline), ((0, 20), ipguard_baseline)]:
        hybrid = {pair_key(row): row for row in groups[split]}
        if set(hybrid) != set(baseline):
            raise ValueError(f"Endpoint {split} model pairs differ from its baseline")
        for pair, row in hybrid.items():
            baseline_row = baseline[pair]
            baseline_label = int(baseline_row["is_positive"].lower() == "true")
            if (
                int(row["pair_label"]) != baseline_label
                or row["attack_type"] != baseline_row["attack_type"]
            ):
                raise ValueError(f"Endpoint {split} metadata differs for {pair}")
            if abs(float(row["score"]) - float(baseline_row["score"])) > 1e-12:
                raise ValueError(f"Endpoint {split} score differs for {pair}")

    summary_rows = []
    for akh_budget, ipguard_budget in EXPECTED_SPLITS:
        group = groups[(akh_budget, ipguard_budget)]
        positive_scores = [
            float(row["score"]) for row in group if int(row["pair_label"]) == 1
        ]
        negative_scores = [
            float(row["score"]) for row in group if int(row["pair_label"]) == 0
        ]
        positive_mean = sum(positive_scores) / len(positive_scores)
        negative_mean = sum(negative_scores) / len(negative_scores)
        tpr, achieved_fpr = operating_point(group)
        summary_rows.append(
            {
                "akh_budget": akh_budget,
                "ipguard_budget": ipguard_budget,
                "akh_ratio": akh_budget / 20,
                "n_pairs": len(group),
                "auc": auc_with_ties(group),
                "tpr_at_5pct_fpr": tpr,
                "achieved_fpr": achieved_fpr,
                "positive_mean_score": positive_mean,
                "negative_mean_score": negative_mean,
                "score_gap": negative_mean - positive_mean,
            }
        )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT.with_suffix(OUTPUT.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        writer.writerows(summary_rows)
    temporary.replace(OUTPUT)
    print(f"Validated 655 rows and exact AKH/IPGuard endpoint parity.")
    print(f"Summary saved to {OUTPUT.resolve()}")


if __name__ == "__main__":
    main()
