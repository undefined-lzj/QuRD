"""Analyze Phase 4A per-query traces without generating queries or running models."""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


BENCHMARK = "SACBenchmark"
DATASET = "CIFAR10"
SEED = 123456789
QUERY_METHODS = ("AKH", "IPGuard")
PREFIX_BUDGETS = (5, 10, 15, 20)
EXPECTED_PAIRS = 131
EXPECTED_POSITIVE = 101
EXPECTED_NEGATIVE = 30
EXPECTED_TRACE_ROWS = 2 * EXPECTED_PAIRS * 20

DEFAULT_INPUT = Path(
    "generated/results/phase4a/"
    "phase4a_query_trace_cifar10_b20_s123456789.csv"
)
DEFAULT_OUTPUT_DIR = Path("generated/results/phase4b")
PREFIX_FILENAME = "phase4b_prefix_scores_cifar10_s123456789.csv"
METRICS_FILENAME = "phase4b_prefix_metrics_cifar10_s123456789.csv"
STABILITY_FILENAME = "phase4b_prefix_stability_cifar10_s123456789.csv"
ATTACK_FILENAME = "phase4b_attack_prefix_metrics_cifar10_s123456789.csv"
REPORT_FILENAME = "phase4b_analysis_report.md"

TRACE_COLUMNS = [
    "benchmark",
    "dataset",
    "seed",
    "source_model",
    "target_model",
    "pair_label",
    "attack_type",
    "query_method",
    "query_index",
    "query_id",
    "source_label",
    "target_label",
    "mismatch",
]
PREFIX_COLUMNS = [
    "source_model",
    "target_model",
    "pair_label",
    "attack_type",
    "seed",
    "akh_score_5",
    "akh_score_10",
    "akh_score_15",
    "akh_score_20",
    "ipguard_score_5",
    "ipguard_score_10",
    "ipguard_score_15",
    "ipguard_score_20",
    "hybrid_15_5",
    "hybrid_10_10",
    "hybrid_5_15",
]
METRIC_COLUMNS = [
    "strategy",
    "n_pairs",
    "n_positive",
    "n_negative",
    "auc",
    "tpr_at_5pct_fpr",
    "actual_fpr",
    "score_threshold",
    "positive_mean_score",
    "negative_mean_score",
    "score_gap",
]
STABILITY_COLUMNS = [
    "query_method",
    "prefix_budget",
    "n_pairs",
    "mean_abs_diff_to_score_20",
    "spearman_vs_score_20",
    "auc",
    "tpr_at_5pct_fpr",
    "actual_fpr",
    "score_threshold",
    "auc_change_vs_5",
    "tpr_change_vs_5",
]
ATTACK_COLUMNS = [
    "query_method",
    "prefix_budget",
    "attack_type",
    "n_positive",
    "n_negative",
    "auc",
]

STRATEGIES = (
    ("AKH-5", "akh_score_5"),
    ("AKH-10", "akh_score_10"),
    ("AKH-15", "akh_score_15"),
    ("AKH-20", "akh_score_20"),
    ("IPGuard-5", "ipguard_score_5"),
    ("IPGuard-10", "ipguard_score_10"),
    ("IPGuard-15", "ipguard_score_15"),
    ("IPGuard-20", "ipguard_score_20"),
    ("Hybrid-15-5", "hybrid_15_5"),
    ("Hybrid-10-10", "hybrid_10_10"),
    ("Hybrid-5-15", "hybrid_5_15"),
)


@dataclass(frozen=True)
class TracePoint:
    query_index: int
    query_id: str
    source_label: int
    target_label: int
    mismatch: int


@dataclass(frozen=True)
class PairMetadata:
    source_model: str
    target_model: str
    pair_label: int
    attack_type: str
    seed: int


@dataclass
class PairTrace:
    metadata: PairMetadata
    methods: dict[str, list[TracePoint]]


def read_and_validate_trace(path: Path) -> dict[tuple[str, str], PairTrace]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing Phase 4A trace CSV: {path}")
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if list(reader.fieldnames or []) != TRACE_COLUMNS:
            raise ValueError(f"Unexpected Phase 4A columns: {reader.fieldnames}")
        raw_rows = list(reader)

    if len(raw_rows) != EXPECTED_TRACE_ROWS:
        raise ValueError(
            f"Expected {EXPECTED_TRACE_ROWS} Phase 4A rows, found {len(raw_rows)}"
        )

    pairs: dict[tuple[str, str], PairTrace] = {}
    seen: set[tuple[str, str, str, int]] = set()
    method_reference: dict[str, dict[int, tuple[str, int]]] = {
        method: {} for method in QUERY_METHODS
    }

    for row in raw_rows:
        if (
            row["benchmark"] != BENCHMARK
            or row["dataset"] != DATASET
            or int(row["seed"]) != SEED
        ):
            raise ValueError("Phase 4A input mixes benchmark, dataset, or seed")
        method = row["query_method"]
        if method not in QUERY_METHODS:
            raise ValueError(f"Unexpected query_method: {method}")

        pair_label = int(row["pair_label"])
        query_index = int(row["query_index"])
        source_label = int(row["source_label"])
        target_label = int(row["target_label"])
        mismatch = int(row["mismatch"])
        if pair_label not in (0, 1):
            raise ValueError(f"Invalid pair_label: {pair_label}")
        if query_index not in range(1, 21):
            raise ValueError(f"Invalid query_index: {query_index}")
        if mismatch not in (0, 1) or mismatch != int(source_label != target_label):
            raise ValueError("mismatch does not equal int(source_label != target_label)")

        pair_key = (row["source_model"], row["target_model"])
        trace_key = (*pair_key, method, query_index)
        if trace_key in seen:
            raise ValueError(f"Duplicate Phase 4A trace key: {trace_key}")
        seen.add(trace_key)

        metadata = PairMetadata(
            source_model=row["source_model"],
            target_model=row["target_model"],
            pair_label=pair_label,
            attack_type=row["attack_type"],
            seed=int(row["seed"]),
        )
        pair = pairs.setdefault(pair_key, PairTrace(metadata=metadata, methods={}))
        if pair.metadata != metadata:
            raise ValueError(f"Metadata changes within model pair: {pair_key}")

        point = TracePoint(
            query_index=query_index,
            query_id=row["query_id"],
            source_label=source_label,
            target_label=target_label,
            mismatch=mismatch,
        )
        pair.methods.setdefault(method, []).append(point)

        reference = method_reference[method]
        query_identity = (point.query_id, point.source_label)
        prior_identity = reference.setdefault(query_index, query_identity)
        if prior_identity != query_identity:
            raise ValueError(
                f"{method} query_id or source_label changes across target models"
            )

    if len(pairs) != EXPECTED_PAIRS:
        raise ValueError(f"Expected {EXPECTED_PAIRS} model pairs, found {len(pairs)}")
    labels = [pair.metadata.pair_label for pair in pairs.values()]
    if sum(labels) != EXPECTED_POSITIVE or labels.count(0) != EXPECTED_NEGATIVE:
        raise ValueError("Phase 4A input does not contain the required 101/30 split")

    expected_indices = list(range(1, 21))
    for method, reference in method_reference.items():
        if sorted(reference) != expected_indices:
            raise ValueError(f"{method} does not have a complete shared query order")
        if len({query_id for query_id, _ in reference.values()}) != 20:
            raise ValueError(f"{method} query_id values are not unique")
    for pair_key, pair in pairs.items():
        if set(pair.methods) != set(QUERY_METHODS):
            raise ValueError(f"Missing query method for pair: {pair_key}")
        for method in QUERY_METHODS:
            pair.methods[method].sort(key=lambda point: point.query_index)
            if [point.query_index for point in pair.methods[method]] != expected_indices:
                raise ValueError(f"Incomplete {method} prefix for pair: {pair_key}")

    if any(
        pair.metadata.attack_type != "unrelated"
        for pair in pairs.values()
        if pair.metadata.pair_label == 0
    ):
        raise ValueError("Every negative pair must use attack_type=unrelated")
    return pairs


def prefix_mean(points: list[TracePoint], budget: int) -> float:
    if budget not in PREFIX_BUDGETS or len(points) != 20:
        raise ValueError("Prefix scores require a complete 20-query trace")
    return sum(point.mismatch for point in points[:budget]) / budget


def build_prefix_rows(pairs: dict[tuple[str, str], PairTrace]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for pair_key in sorted(pairs):
        pair = pairs[pair_key]
        akh = pair.methods["AKH"]
        ipguard = pair.methods["IPGuard"]
        row: dict[str, Any] = {
            "source_model": pair.metadata.source_model,
            "target_model": pair.metadata.target_model,
            "pair_label": pair.metadata.pair_label,
            "attack_type": pair.metadata.attack_type,
            "seed": pair.metadata.seed,
        }
        for budget in PREFIX_BUDGETS:
            row[f"akh_score_{budget}"] = prefix_mean(akh, budget)
            row[f"ipguard_score_{budget}"] = prefix_mean(ipguard, budget)

        row["hybrid_15_5"] = (
            sum(point.mismatch for point in akh[:15])
            + sum(point.mismatch for point in ipguard[:5])
        ) / 20
        row["hybrid_10_10"] = (
            sum(point.mismatch for point in akh[:10])
            + sum(point.mismatch for point in ipguard[:10])
        ) / 20
        row["hybrid_5_15"] = (
            sum(point.mismatch for point in akh[:5])
            + sum(point.mismatch for point in ipguard[:15])
        ) / 20
        rows.append(row)
    return rows


def auc_lower_is_positive(labels: list[int], scores: list[float]) -> float:
    """Tie-aware AUC using -score as the positive decision value."""
    if len(labels) != len(scores) or not labels:
        raise ValueError("AUC labels and scores must be non-empty and aligned")
    n_positive = sum(labels)
    n_negative = len(labels) - n_positive
    if n_positive == 0 or n_negative == 0:
        raise ValueError("AUC requires both positive and negative examples")

    ranked = sorted(zip(labels, (-score for score in scores)), key=lambda item: item[1])
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
    return (
        positive_rank_sum - n_positive * (n_positive + 1) / 2
    ) / (n_positive * n_negative)


def operating_point_lower_score(
    labels: list[int], scores: list[float], fpr_limit: float = 0.05
) -> tuple[float, float, float]:
    """Return TPR, achieved FPR and threshold for the rule score <= threshold."""
    if len(labels) != len(scores) or not labels:
        raise ValueError("Operating-point labels and scores must be aligned")
    n_positive = sum(labels)
    n_negative = len(labels) - n_positive
    if n_positive == 0 or n_negative == 0:
        raise ValueError("Operating point requires both classes")

    best_tpr = 0.0
    best_fpr = 0.0
    best_threshold = -math.inf
    for threshold in [-math.inf, *sorted(set(scores))]:
        true_positive = sum(
            label == 1 and score <= threshold
            for label, score in zip(labels, scores)
        )
        false_positive = sum(
            label == 0 and score <= threshold
            for label, score in zip(labels, scores)
        )
        tpr = true_positive / n_positive
        fpr = false_positive / n_negative
        if fpr <= fpr_limit and tpr > best_tpr:
            best_tpr = tpr
            best_fpr = fpr
            best_threshold = threshold
    return best_tpr, best_fpr, best_threshold


def summarize_scores(
    strategy: str, labels: list[int], scores: list[float]
) -> dict[str, Any]:
    positive_scores = [score for label, score in zip(labels, scores) if label == 1]
    negative_scores = [score for label, score in zip(labels, scores) if label == 0]
    tpr, actual_fpr, threshold = operating_point_lower_score(labels, scores)
    positive_mean = sum(positive_scores) / len(positive_scores)
    negative_mean = sum(negative_scores) / len(negative_scores)
    return {
        "strategy": strategy,
        "n_pairs": len(labels),
        "n_positive": len(positive_scores),
        "n_negative": len(negative_scores),
        "auc": auc_lower_is_positive(labels, scores),
        "tpr_at_5pct_fpr": tpr,
        "actual_fpr": actual_fpr,
        "score_threshold": threshold,
        "positive_mean_score": positive_mean,
        "negative_mean_score": negative_mean,
        "score_gap": negative_mean - positive_mean,
    }


def build_metric_rows(prefix_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    labels = [int(row["pair_label"]) for row in prefix_rows]
    return [
        summarize_scores(
            strategy,
            labels,
            [float(row[column]) for row in prefix_rows],
        )
        for strategy, column in STRATEGIES
    ]


def average_ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        rank = ((start + 1) + end) / 2
        for position in range(start, end):
            ranks[order[position]] = rank
        start = end
    return ranks


def spearman_with_ties(left: list[float], right: list[float]) -> float | None:
    if len(left) != len(right) or not left:
        raise ValueError("Spearman inputs must be non-empty and aligned")
    left_ranks = average_ranks(left)
    right_ranks = average_ranks(right)
    left_mean = sum(left_ranks) / len(left_ranks)
    right_mean = sum(right_ranks) / len(right_ranks)
    numerator = sum(
        (left_rank - left_mean) * (right_rank - right_mean)
        for left_rank, right_rank in zip(left_ranks, right_ranks)
    )
    left_ss = sum((rank - left_mean) ** 2 for rank in left_ranks)
    right_ss = sum((rank - right_mean) ** 2 for rank in right_ranks)
    denominator = math.sqrt(left_ss * right_ss)
    return None if denominator == 0 else numerator / denominator


def build_stability_rows(
    prefix_rows: list[dict[str, Any]], metric_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    metric_index = {row["strategy"]: row for row in metric_rows}
    rows: list[dict[str, Any]] = []
    for method in QUERY_METHODS:
        prefix = method.lower()
        final_scores = [float(row[f"{prefix}_score_20"]) for row in prefix_rows]
        baseline = metric_index[f"{method}-5"]
        for budget in PREFIX_BUDGETS:
            scores = [float(row[f"{prefix}_score_{budget}"]) for row in prefix_rows]
            metric = metric_index[f"{method}-{budget}"]
            rows.append(
                {
                    "query_method": method,
                    "prefix_budget": budget,
                    "n_pairs": len(scores),
                    "mean_abs_diff_to_score_20": sum(
                        abs(score - final)
                        for score, final in zip(scores, final_scores)
                    )
                    / len(scores),
                    "spearman_vs_score_20": spearman_with_ties(scores, final_scores),
                    "auc": metric["auc"],
                    "tpr_at_5pct_fpr": metric["tpr_at_5pct_fpr"],
                    "actual_fpr": metric["actual_fpr"],
                    "score_threshold": metric["score_threshold"],
                    "auc_change_vs_5": metric["auc"] - baseline["auc"],
                    "tpr_change_vs_5": (
                        metric["tpr_at_5pct_fpr"]
                        - baseline["tpr_at_5pct_fpr"]
                    ),
                }
            )
    return rows


def build_attack_rows(prefix_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    attack_types = sorted(
        {
            row["attack_type"]
            for row in prefix_rows
            if int(row["pair_label"]) == 1
        }
    )
    negatives = [row for row in prefix_rows if int(row["pair_label"]) == 0]
    if not negatives or any(row["attack_type"] != "unrelated" for row in negatives):
        raise ValueError("Attack analysis requires unrelated negative pairs")

    rows: list[dict[str, Any]] = []
    for method in QUERY_METHODS:
        prefix = method.lower()
        for budget in PREFIX_BUDGETS:
            score_column = f"{prefix}_score_{budget}"
            for attack_type in attack_types:
                positives = [
                    row
                    for row in prefix_rows
                    if int(row["pair_label"]) == 1
                    and row["attack_type"] == attack_type
                ]
                group = positives + negatives
                labels = [int(row["pair_label"]) for row in group]
                scores = [float(row[score_column]) for row in group]
                rows.append(
                    {
                        "query_method": method,
                        "prefix_budget": budget,
                        "attack_type": attack_type,
                        "n_positive": len(positives),
                        "n_negative": len(negatives),
                        "auc": auc_lower_is_positive(labels, scores),
                    }
                )
    return rows


def format_number(value: float | None) -> str:
    if value is None:
        return "n.a."
    return f"{value:.4f}"


def join_attack_changes(changes: list[tuple[str, float]], language: str) -> str:
    improving = [(name, change) for name, change in changes if change > 1e-12]
    if not improving:
        return "无" if language == "zh" else "none"
    return ", ".join(f"{name} ({change:+.4f})" for name, change in improving)


def build_report(
    metric_rows: list[dict[str, Any]],
    stability_rows: list[dict[str, Any]],
    attack_rows: list[dict[str, Any]],
) -> str:
    metrics = {row["strategy"]: row for row in metric_rows}
    stability = {
        (row["query_method"], int(row["prefix_budget"])): row
        for row in stability_rows
    }
    attack = {
        (row["query_method"], int(row["prefix_budget"]), row["attack_type"]): row
        for row in attack_rows
    }

    akh_5 = stability[("AKH", 5)]
    ipguard_5 = stability[("IPGuard", 5)]
    akh_rho_key = (
        -math.inf
        if akh_5["spearman_vs_score_20"] is None
        else akh_5["spearman_vs_score_20"]
    )
    ipguard_rho_key = (
        -math.inf
        if ipguard_5["spearman_vs_score_20"] is None
        else ipguard_5["spearman_vs_score_20"]
    )
    akh_mad = akh_5["mean_abs_diff_to_score_20"]
    ipguard_mad = ipguard_5["mean_abs_diff_to_score_20"]
    if akh_mad == ipguard_mad and akh_rho_key == ipguard_rho_key:
        stable_zh = "两种方法在两项稳定性指标上相同。"
        stable_en = "The two methods are equal on both stability measures."
    elif akh_mad <= ipguard_mad and akh_rho_key >= ipguard_rho_key:
        stable_zh = "AKH在两项稳定性指标上都更好。"
        stable_en = "AKH is more stable on both stability measures."
    elif ipguard_mad <= akh_mad and ipguard_rho_key >= akh_rho_key:
        stable_zh = "IPGuard在两项稳定性指标上都更好。"
        stable_en = "IPGuard is more stable on both stability measures."
    else:
        stable_zh = "两项稳定性指标给出的结论不一致，不能判定单一赢家。"
        stable_en = "The two stability measures disagree, so there is no single winner."

    improvements: dict[str, tuple[float, float]] = {}
    for method in QUERY_METHODS:
        improvements[method] = (
            metrics[f"{method}-20"]["auc"] - metrics[f"{method}-5"]["auc"],
            metrics[f"{method}-20"]["tpr_at_5pct_fpr"]
            - metrics[f"{method}-5"]["tpr_at_5pct_fpr"],
        )
    if improvements["AKH"] > improvements["IPGuard"]:
        improvement_zh = "按AUC增量优先、TPR增量次优的规则，AKH提升更明显。"
        improvement_en = "Using AUC gain first and TPR gain second, AKH improves more."
    elif improvements["IPGuard"] > improvements["AKH"]:
        improvement_zh = "按AUC增量优先、TPR增量次优的规则，IPGuard提升更明显。"
        improvement_en = (
            "Using AUC gain first and TPR gain second, IPGuard improves more."
        )
    else:
        improvement_zh = "两种方法从5次到20次的提升相同。"
        improvement_en = "The two methods improve equally from 5 to 20 queries."

    rho_akh = akh_5["spearman_vs_score_20"]
    rho_ipguard = ipguard_5["spearman_vs_score_20"]
    predictable = [
        method
        for method, rho in [("AKH", rho_akh), ("IPGuard", rho_ipguard)]
        if rho is not None and rho >= 0.7
    ]
    if len(predictable) == 2:
        predict_zh = "两种方法的前5次分数都能较好反映最终排序。"
        predict_en = "For both methods, five-query scores track the final ranking well."
    elif predictable:
        predict_zh = f"仅{predictable[0]}的前5次分数能较好反映最终排序。"
        predict_en = (
            f"Only {predictable[0]} has a five-query score that tracks the final "
            "ranking well."
        )
    else:
        predict_zh = "前5次分数不足以稳定反映最终排序。"
        predict_en = "Five-query scores do not reliably track the final ranking."

    attack_types = sorted({key[2] for key in attack})
    akh_changes = sorted(
        [
            (
                attack_type,
                attack[("AKH", 20, attack_type)]["auc"]
                - attack[("AKH", 5, attack_type)]["auc"],
            )
            for attack_type in attack_types
        ],
        key=lambda item: item[1],
        reverse=True,
    )
    ipguard_dependent = sorted(
        [
            (
                attack_type,
                attack[("IPGuard", 20, attack_type)]["auc"]
                - attack[("AKH", 20, attack_type)]["auc"],
            )
            for attack_type in attack_types
            if attack[("IPGuard", 20, attack_type)]["auc"]
            > attack[("AKH", 20, attack_type)]["auc"] + 1e-12
        ],
        key=lambda item: item[1],
        reverse=True,
    )
    ipguard_text = (
        ", ".join(f"{name} ({gap:+.4f})" for name, gap in ipguard_dependent)
        or "无"
    )
    ipguard_text_en = (
        ", ".join(f"{name} ({gap:+.4f})" for name, gap in ipguard_dependent)
        or "none"
    )

    hybrid_names = ("Hybrid-15-5", "Hybrid-10-10", "Hybrid-5-15")
    best_hybrid = max(
        hybrid_names,
        key=lambda name: (
            metrics[name]["auc"],
            metrics[name]["tpr_at_5pct_fpr"],
            metrics[name]["score_gap"],
        ),
    )
    best = metrics[best_hybrid]
    more_stable_method = min(
        QUERY_METHODS,
        key=lambda method: stability[(method, 5)]["mean_abs_diff_to_score_20"],
    )

    return f"""# 阶段4B：逐查询证据变化分析

## 中文结论

1. **前5次查询稳定性。** {stable_zh} AKH的平均绝对差为{format_number(akh_5['mean_abs_diff_to_score_20'])}、Spearman相关系数为{format_number(rho_akh)}；IPGuard分别为{format_number(ipguard_5['mean_abs_diff_to_score_20'])}和{format_number(rho_ipguard)}。
2. **从5次增加到20次的提升。** {improvement_zh} AKH的AUC/TPR变化为{improvements['AKH'][0]:+.4f}/{improvements['AKH'][1]:+.4f}；IPGuard为{improvements['IPGuard'][0]:+.4f}/{improvements['IPGuard'][1]:+.4f}。
3. **前5次能否预测最终score。** {predict_zh} 这里采用Spearman不低于0.7作为描述性判断标准，不是最终检测阈值。
4. **应继续增加AKH查询的攻击类型。** 按AKH从5次到20次的AUC增量大于0筛选：{join_attack_changes(akh_changes, 'zh')}。该结论仅用于离线分析。
5. **更依赖IPGuard的攻击类型。** 按20次查询时IPGuard AUC高于AKH筛选：{ipguard_text}。
6. **三种固定分配中的最佳方案。** 按AUC优先、TPR@5%FPR次优、正负均值差再次优选择{best_hybrid}，AUC={best['auc']:.4f}，TPR={best['tpr_at_5pct_fpr']:.4f}，实际FPR={best['actual_fpr']:.4f}。
7. **对动态预算规则的启示。** 按前5次与最终分数的平均绝对差，后续规则可以先使用{more_stable_method}收集初始证据，再根据可观测的前缀分数变化和AKH/IPGuard分歧决定是否追加查询。pair_label和attack_type只能用于离线评估，不能作为运行时输入；本报告阈值也不能直接作为最终测试阈值。

## English conclusions

1. **Stability after five queries.** {stable_en} AKH has a mean absolute difference of {format_number(akh_5['mean_abs_diff_to_score_20'])} and a Spearman correlation of {format_number(rho_akh)}; IPGuard has {format_number(ipguard_5['mean_abs_diff_to_score_20'])} and {format_number(rho_ipguard)}, respectively.
2. **Improvement from 5 to 20 queries.** {improvement_en} The AUC/TPR changes are {improvements['AKH'][0]:+.4f}/{improvements['AKH'][1]:+.4f} for AKH and {improvements['IPGuard'][0]:+.4f}/{improvements['IPGuard'][1]:+.4f} for IPGuard.
3. **Can five queries predict the final score?** {predict_en} A Spearman correlation of at least 0.7 is used only as a descriptive rule, not as a final detection threshold.
4. **Attack types that benefit from more AKH queries.** Positive AKH AUC gains from 5 to 20 queries: {join_attack_changes(akh_changes, 'en')}. This is an offline evaluation result only.
5. **Attack types more dependent on IPGuard.** IPGuard AUC exceeds AKH AUC at 20 queries for: {ipguard_text_en}.
6. **Best fixed allocation.** Ranking by AUC first, TPR@5%FPR second, and the mean-score gap third selects {best_hybrid}, with AUC={best['auc']:.4f}, TPR={best['tpr_at_5pct_fpr']:.4f}, and achieved FPR={best['actual_fpr']:.4f}.
7. **Implication for a future dynamic rule.** Based on the mean absolute difference between five-query and final scores, a future rule could begin with {more_stable_method} and use observable prefix-score changes and AKH/IPGuard disagreement to decide whether to request more evidence. pair_label and attack_type must remain offline evaluation metadata, and development thresholds in this report must not become final test thresholds.
"""


def validate_outputs(
    prefix_rows: list[dict[str, Any]],
    metric_rows: list[dict[str, Any]],
    stability_rows: list[dict[str, Any]],
    attack_rows: list[dict[str, Any]],
) -> None:
    if len(prefix_rows) != EXPECTED_PAIRS:
        raise ValueError("Prefix score output must contain 131 rows")
    if any(set(row) != set(PREFIX_COLUMNS) for row in prefix_rows):
        raise ValueError("Prefix score output columns are incomplete")
    if {row["strategy"] for row in metric_rows} != {
        strategy for strategy, _ in STRATEGIES
    }:
        raise ValueError("Metric output does not contain all 11 strategies")
    if len(metric_rows) != len(STRATEGIES):
        raise ValueError("Metric output contains duplicate strategies")
    for row in metric_rows:
        if (
            row["n_pairs"] != EXPECTED_PAIRS
            or row["n_positive"] != EXPECTED_POSITIVE
            or row["n_negative"] != EXPECTED_NEGATIVE
        ):
            raise ValueError("Metric output has incorrect class counts")
        for column in ("auc", "tpr_at_5pct_fpr", "actual_fpr"):
            if not 0 <= float(row[column]) <= 1:
                raise ValueError(f"Invalid metric value in {column}")
    if len(stability_rows) != len(QUERY_METHODS) * len(PREFIX_BUDGETS):
        raise ValueError("Stability output must contain eight method-prefix rows")
    if {
        (row["query_method"], int(row["prefix_budget"]))
        for row in stability_rows
    } != {
        (method, budget)
        for method in QUERY_METHODS
        for budget in PREFIX_BUDGETS
    }:
        raise ValueError("Stability output has incomplete method-prefix keys")
    attack_types = {
        row["attack_type"]
        for row in prefix_rows
        if int(row["pair_label"]) == 1
    }
    if len(attack_rows) != len(QUERY_METHODS) * len(PREFIX_BUDGETS) * len(attack_types):
        raise ValueError("Attack-prefix output has an unexpected number of rows")
    if any(int(row["n_negative"]) != EXPECTED_NEGATIVE for row in attack_rows):
        raise ValueError("Attack-prefix output must use all 30 unrelated negatives")
    for row in prefix_rows:
        for column in PREFIX_COLUMNS[5:]:
            if not 0 <= float(row[column]) <= 1:
                raise ValueError(f"Invalid prefix or hybrid score in {column}")
        if abs(
            row["hybrid_15_5"]
            - (15 * row["akh_score_15"] + 5 * row["ipguard_score_5"]) / 20
        ) > 1e-12:
            raise ValueError("hybrid_15_5 is inconsistent with its prefixes")
        if abs(
            row["hybrid_10_10"]
            - (10 * row["akh_score_10"] + 10 * row["ipguard_score_10"]) / 20
        ) > 1e-12:
            raise ValueError("hybrid_10_10 is inconsistent with its prefixes")
        if abs(
            row["hybrid_5_15"]
            - (5 * row["akh_score_5"] + 15 * row["ipguard_score_15"]) / 20
        ) > 1e-12:
            raise ValueError("hybrid_5_15 is inconsistent with its prefixes")


def write_csv(path: Path, columns: list[str], rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Validate the Phase 4A trace without creating Phase 4B outputs.",
    )
    args = parser.parse_args()

    pairs = read_and_validate_trace(args.input)
    if args.preflight_only:
        print(
            "Phase 4B preflight passed: 5240 trace rows, 131 pairs, "
            "101 positives, 30 unrelated negatives, and complete AKH/IPGuard prefixes."
        )
        return

    prefix_rows = build_prefix_rows(pairs)
    metric_rows = build_metric_rows(prefix_rows)
    stability_rows = build_stability_rows(prefix_rows, metric_rows)
    attack_rows = build_attack_rows(prefix_rows)
    validate_outputs(prefix_rows, metric_rows, stability_rows, attack_rows)
    report = build_report(metric_rows, stability_rows, attack_rows)

    outputs = {
        PREFIX_FILENAME: (PREFIX_COLUMNS, prefix_rows),
        METRICS_FILENAME: (METRIC_COLUMNS, metric_rows),
        STABILITY_FILENAME: (STABILITY_COLUMNS, stability_rows),
        ATTACK_FILENAME: (ATTACK_COLUMNS, attack_rows),
    }
    for filename, (columns, rows) in outputs.items():
        write_csv(args.output_dir / filename, columns, rows)
    write_text(args.output_dir / REPORT_FILENAME, report)
    print(f"Saved four validated CSV files and one report to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
