"""Export Phase 4A per-query hard-label traces from Phase 3 endpoint caches."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


BENCHMARK = "SACBenchmark"
DATASET = "CIFAR10"
SEED = 123456789
BUDGET = 20
EXPECTED_PAIRS = 131
EXPECTED_POSITIVE = 101
EXPECTED_NEGATIVE = 30
EXPECTED_ROWS = 2 * EXPECTED_PAIRS * BUDGET
SOURCE_MODEL = "train(vgg_model,CIFAR10,base)"
DEFAULT_CACHE_ROOT = Path("generated/phase3_fixed_hybrid_cache")
DEFAULT_PHASE3_SCORES = Path(
    "generated/results/phase3/"
    "phase3_fixed_hybrid_cifar10_b20_s123456789.csv"
)
DEFAULT_OUTPUT = Path(
    "generated/results/phase4a/"
    "phase4a_query_trace_cifar10_b20_s123456789.csv"
)
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


@dataclass(frozen=True)
class EndpointSpec:
    method: str
    ratio_dir: str
    budget_dir: str
    score_column: str


ENDPOINTS = (
    EndpointSpec("AKH", "akh_ratio=1.00", "akh_budget=20,ipguard_budget=0", "akh_score"),
    EndpointSpec(
        "IPGuard",
        "akh_ratio=0.00",
        "akh_budget=0,ipguard_budget=20",
        "ipguard_score",
    ),
)


def load_tensor(path: Path, expected_length: int = BUDGET) -> torch.Tensor:
    if not path.is_file():
        raise FileNotFoundError(f"Missing Phase 3 cache file: {path}")
    with path.open("rb") as handle:
        value = pickle.load(handle)
    if not isinstance(value, torch.Tensor) or len(value) != expected_length:
        raise ValueError(
            f"{path} must contain a tensor with {expected_length} entries"
        )
    return value.detach().cpu()


def stable_query_id(method: str, query: torch.Tensor) -> str:
    """Identify a query by method plus a stable hash of shape, dtype and bytes."""
    query = query.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(b"phase4a-query-v1\0")
    digest.update(method.encode("utf-8"))
    digest.update(str(query.dtype).encode("ascii"))
    digest.update(json.dumps(list(query.shape), separators=(",", ":")).encode("ascii"))
    digest.update(query.numpy().tobytes(order="C"))
    return f"{method.lower()}-{digest.hexdigest()}"


def endpoint_cache_dir(cache_root: Path, endpoint: EndpointSpec) -> Path:
    return (
        cache_root
        / BENCHMARK
        / DATASET
        / f"seed={SEED}"
        / "method=Fixed-Hybrid"
        / f"budget={BUDGET}"
        / endpoint.ratio_dir
        / endpoint.budget_dir
        / SOURCE_MODEL
    )


def read_phase3_endpoints(
    path: Path,
) -> dict[str, dict[tuple[str, str], dict[str, str]]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing Phase 3 score file: {path}")
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))

    result: dict[str, dict[tuple[str, str], dict[str, str]]] = {}
    for endpoint in ENDPOINTS:
        expected_akh_budget = 20 if endpoint.method == "AKH" else 0
        selected = [
            row
            for row in rows
            if int(row["akh_budget"]) == expected_akh_budget
            and int(row["ipguard_budget"]) == BUDGET - expected_akh_budget
        ]
        for row in selected:
            if (
                row["method"] != "Fixed-Hybrid"
                or int(row["budget"]) != BUDGET
                or int(row["seed"]) != SEED
                or row["dataset"] != DATASET
                or row["source_model"] != SOURCE_MODEL
                or int(row["pair_label"]) not in (0, 1)
            ):
                raise ValueError(
                    f"Phase 3 {endpoint.method} endpoint mixes configurations"
                )
        index = {
            (row["source_model"], row["target_model"]): row for row in selected
        }
        if len(selected) != EXPECTED_PAIRS or len(index) != EXPECTED_PAIRS:
            raise ValueError(
                f"Phase 3 {endpoint.method} endpoint must contain "
                f"{EXPECTED_PAIRS} unique model pairs"
            )
        result[endpoint.method] = index

    if set(result["AKH"]) != set(result["IPGuard"]):
        raise ValueError("Phase 3 endpoint model-pair sets differ")
    for pair in result["AKH"]:
        akh = result["AKH"][pair]
        ipguard = result["IPGuard"][pair]
        if (
            akh["pair_label"] != ipguard["pair_label"]
            or akh["attack_type"] != ipguard["attack_type"]
        ):
            raise ValueError(f"Phase 3 endpoint metadata differs for {pair}")
    return result


def preflight_inputs(
    cache_root: Path,
    endpoint_rows: dict[str, dict[tuple[str, str], dict[str, str]]],
) -> None:
    for endpoint in ENDPOINTS:
        cache_dir = endpoint_cache_dir(cache_root, endpoint)
        load_tensor(cache_dir / "queries.pickle")
        load_tensor(cache_dir / "HardLabels()" / "source.pickle")
        for source_model, target_model in endpoint_rows[endpoint.method]:
            if target_model == source_model:
                continue
            target_path = (
                cache_dir
                / "HardLabels()"
                / "targets"
                / f"{target_model}.pickle"
            )
            if not target_path.is_file():
                raise FileNotFoundError(
                    f"Missing {endpoint.method} target-label cache: {target_path}"
                )


def rows_for_pair(
    *,
    target_model: str,
    pair_label: int,
    attack_type: str,
    query_method: str,
    query_ids: list[str],
    source_labels: torch.Tensor,
    target_labels: torch.Tensor,
) -> list[dict[str, Any]]:
    if not (len(query_ids) == len(source_labels) == len(target_labels) == BUDGET):
        raise ValueError("Every Phase 4A method must provide exactly 20 query responses")
    records = []
    for index, (query_id, source_label, target_label) in enumerate(
        zip(query_ids, source_labels.tolist(), target_labels.tolist()), start=1
    ):
        source_label = int(source_label)
        target_label = int(target_label)
        records.append(
            {
                "benchmark": BENCHMARK,
                "dataset": DATASET,
                "seed": SEED,
                "source_model": SOURCE_MODEL,
                "target_model": target_model,
                "pair_label": pair_label,
                "attack_type": attack_type,
                "query_method": query_method,
                "query_index": index,
                "query_id": query_id,
                "source_label": source_label,
                "target_label": target_label,
                "mismatch": int(source_label != target_label),
            }
        )
    return records


def build_trace_rows(
    cache_root: Path,
    endpoint_rows: dict[str, dict[tuple[str, str], dict[str, str]]],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    pairs = sorted(endpoint_rows["AKH"])

    for endpoint in ENDPOINTS:
        cache_dir = endpoint_cache_dir(cache_root, endpoint)
        queries = load_tensor(cache_dir / "queries.pickle")
        source_labels = load_tensor(cache_dir / "HardLabels()" / "source.pickle")
        query_ids = [stable_query_id(endpoint.method, query) for query in queries]
        if len(set(query_ids)) != BUDGET:
            raise ValueError(f"{endpoint.method} query pool contains duplicate query IDs")

        for source_model, target_model in pairs:
            metadata = endpoint_rows[endpoint.method][(source_model, target_model)]
            if target_model == source_model:
                target_labels = source_labels
            else:
                target_labels = load_tensor(
                    cache_dir
                    / "HardLabels()"
                    / "targets"
                    / f"{target_model}.pickle"
                )
            records.extend(
                rows_for_pair(
                    target_model=target_model,
                    pair_label=int(metadata["pair_label"]),
                    attack_type=metadata["attack_type"],
                    query_method=endpoint.method,
                    query_ids=query_ids,
                    source_labels=source_labels,
                    target_labels=target_labels,
                )
            )
    return records


def validate_trace_rows(
    rows: list[dict[str, Any]],
    endpoint_rows: dict[str, dict[tuple[str, str], dict[str, str]]],
) -> None:
    if len(rows) != EXPECTED_ROWS:
        raise ValueError(f"Expected {EXPECTED_ROWS} trace rows, found {len(rows)}")

    pairs = {(row["source_model"], row["target_model"]) for row in rows}
    if len(pairs) != EXPECTED_PAIRS:
        raise ValueError(f"Expected {EXPECTED_PAIRS} model pairs, found {len(pairs)}")
    pair_metadata = {
        pair: {
            (int(row["pair_label"]), row["attack_type"])
            for row in rows
            if (row["source_model"], row["target_model"]) == pair
        }
        for pair in pairs
    }
    if any(len(values) != 1 for values in pair_metadata.values()):
        raise ValueError("pair_label or attack_type changes within a model pair")
    positives = sum(next(iter(values))[0] for values in pair_metadata.values())
    if positives != EXPECTED_POSITIVE or len(pairs) - positives != EXPECTED_NEGATIVE:
        raise ValueError("Trace does not contain the required 101/30 pair split")

    seen_keys: set[tuple[str, str, str, int]] = set()
    reference_order: dict[str, dict[int, str]] = {}
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        if (
            row["benchmark"] != BENCHMARK
            or row["dataset"] != DATASET
            or int(row["seed"]) != SEED
        ):
            raise ValueError("Trace mixes benchmark, dataset, or seed")
        if row["query_method"] not in {"AKH", "IPGuard"}:
            raise ValueError(f"Unexpected query method: {row['query_method']}")
        index = int(row["query_index"])
        key = (
            row["source_model"],
            row["target_model"],
            row["query_method"],
            index,
        )
        if key in seen_keys:
            raise ValueError(f"Duplicate trace key: {key}")
        seen_keys.add(key)
        expected_mismatch = int(int(row["source_label"]) != int(row["target_label"]))
        if int(row["mismatch"]) != expected_mismatch:
            raise ValueError(f"Incorrect mismatch for {key}")
        method_order = reference_order.setdefault(row["query_method"], {})
        previous_id = method_order.setdefault(index, row["query_id"])
        if previous_id != row["query_id"]:
            raise ValueError(
                f"Query order or ID changes across targets for {row['query_method']}"
            )
        grouped.setdefault(key[:3], []).append(row)

    expected_indices = set(range(1, BUDGET + 1))
    for method, order in reference_order.items():
        if set(order) != expected_indices or len(set(order.values())) != BUDGET:
            raise ValueError(f"{method} query indices or IDs are incomplete")
    if len(grouped) != 2 * EXPECTED_PAIRS:
        raise ValueError("Every model pair must appear once under each query method")

    for (source_model, target_model, method), group in grouped.items():
        indices = {int(row["query_index"]) for row in group}
        if len(group) != BUDGET or indices != expected_indices:
            raise ValueError(
                f"{method} trace for {(source_model, target_model)} is incomplete"
            )
        mean_mismatch = sum(int(row["mismatch"]) for row in group) / BUDGET
        endpoint = endpoint_rows[method][(source_model, target_model)]
        score_column = next(
            endpoint_spec.score_column
            for endpoint_spec in ENDPOINTS
            if endpoint_spec.method == method
        )
        if abs(mean_mismatch - float(endpoint[score_column])) > 1e-12:
            raise ValueError(
                f"{method} mismatch mean does not reproduce Phase 3 for "
                f"{(source_model, target_model)}"
            )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=TRACE_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--phase3-scores", type=Path, default=DEFAULT_PHASE3_SCORES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Check Phase 3 endpoints and cache coverage without exporting rows.",
    )
    args = parser.parse_args()

    endpoint_rows = read_phase3_endpoints(args.phase3_scores)
    preflight_inputs(args.cache_root, endpoint_rows)
    if args.preflight_only:
        print(
            "Phase 4A preflight passed: 131 pairs and both 20-query endpoint "
            "label caches are complete."
        )
        return
    rows = build_trace_rows(args.cache_root, endpoint_rows)
    validate_trace_rows(rows, endpoint_rows)
    write_csv(args.output, rows)
    print(
        f"Saved {len(rows)} validated per-query rows to {args.output.resolve()}"
    )


if __name__ == "__main__":
    main()
