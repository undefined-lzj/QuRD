from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import torch


FIXED_HYBRID_METHOD = "Fixed-Hybrid"
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
FIXED_HYBRID_KEY_COLUMNS = [
    "dataset",
    "method",
    "budget",
    "seed",
    "source_model",
    "target_model",
    "akh_budget",
    "ipguard_budget",
]


def _optional_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


class FixedHybridScoresCsv:
    """Persist all hybrid ratios without allowing one ratio to overwrite another."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._records: dict[tuple[str, ...], dict[str, Any]] = {}
        if path.is_file():
            with path.open(newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                if reader.fieldnames != FIXED_HYBRID_COLUMNS:
                    raise ValueError(
                        f"Unexpected Fixed-Hybrid CSV columns in {path}: "
                        f"{reader.fieldnames}"
                    )
                for record in reader:
                    normalized = self._normalize(record)
                    self._records[self._key(normalized)] = normalized

    def upsert(self, record: dict[str, Any]) -> None:
        normalized = self._normalize(record)
        self._records[self._key(normalized)] = normalized
        self._flush()

    @staticmethod
    def _normalize(record: dict[str, Any]) -> dict[str, Any]:
        if str(record["method"]) != FIXED_HYBRID_METHOD:
            raise ValueError(f"method must be {FIXED_HYBRID_METHOD}")
        budget = int(record["budget"])
        akh_budget = int(record["akh_budget"])
        ipguard_budget = int(record["ipguard_budget"])
        if (
            akh_budget < 0
            or ipguard_budget < 0
            or akh_budget + ipguard_budget != budget
        ):
            raise ValueError(
                "AKH and IPGuard budgets must be non-negative and sum to budget"
            )

        score = float(record["score"])
        hybrid_score = float(record.get("hybrid_score", score))
        if (
            not math.isfinite(score)
            or not 0.0 <= score <= 1.0
            or abs(score - hybrid_score) > 1e-12
        ):
            raise ValueError("score must equal hybrid_score")

        akh_score = _optional_float(record.get("akh_score"))
        ipguard_score = _optional_float(record.get("ipguard_score"))
        if akh_budget == 0 and akh_score is not None:
            raise ValueError("akh_score must be blank when akh_budget is zero")
        if ipguard_budget == 0 and ipguard_score is not None:
            raise ValueError(
                "ipguard_score must be blank when ipguard_budget is zero"
            )
        if akh_budget > 0 and akh_score is None:
            raise ValueError("akh_score is required when akh_budget is positive")
        if ipguard_budget > 0 and ipguard_score is None:
            raise ValueError(
                "ipguard_score is required when ipguard_budget is positive"
            )
        component_score = (
            (akh_score or 0.0) * akh_budget
            + (ipguard_score or 0.0) * ipguard_budget
        ) / budget
        if abs(score - component_score) > 1e-12:
            raise ValueError("hybrid_score must be the weighted component score")

        pair_label = int(record["pair_label"])
        if pair_label not in (0, 1):
            raise ValueError("pair_label must be 0 or 1")

        return {
            "method": FIXED_HYBRID_METHOD,
            "budget": budget,
            "seed": int(record["seed"]),
            "source_model": str(record["source_model"]),
            "target_model": str(record["target_model"]),
            "score": score,
            "dataset": str(record["dataset"]),
            "pair_label": pair_label,
            "attack_type": str(record["attack_type"]),
            "akh_budget": akh_budget,
            "ipguard_budget": ipguard_budget,
            "akh_ratio": akh_budget / budget,
            "akh_score": akh_score,
            "ipguard_score": ipguard_score,
            "hybrid_score": hybrid_score,
        }

    @staticmethod
    def _key(record: dict[str, Any]) -> tuple[str, ...]:
        return tuple(str(record[column]) for column in FIXED_HYBRID_KEY_COLUMNS)

    def _flush(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = self.path.with_suffix(self.path.suffix + ".tmp")
        with temporary_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIXED_HYBRID_COLUMNS)
            writer.writeheader()
            writer.writerows(
                sorted(
                    self._records.values(),
                    key=lambda row: (
                        row["akh_budget"],
                        row["source_model"],
                        row["target_model"],
                    ),
                )
            )
        temporary_path.replace(self.path)


def make_fixed_hybrid_queries(
    akh_pool: torch.Tensor,
    ipguard_pool: torch.Tensor,
    akh_budget: int,
    ipguard_budget: int,
) -> torch.Tensor:
    """Take fixed prefixes from source-only baseline query pools."""
    if akh_budget < 0 or ipguard_budget < 0:
        raise ValueError("Component budgets must be non-negative")
    if len(akh_pool) < akh_budget or len(ipguard_pool) < ipguard_budget:
        raise ValueError("A baseline query pool is smaller than its requested budget")

    parts = []
    if akh_budget:
        parts.append(akh_pool[:akh_budget])
    if ipguard_budget:
        parts.append(ipguard_pool[:ipguard_budget])
    if not parts:
        raise ValueError("The total hybrid budget must be positive")
    return torch.cat(parts, dim=0)


def score_fixed_hybrid_labels(
    source_labels: torch.Tensor,
    target_labels: torch.Tensor,
    akh_budget: int,
    ipguard_budget: int,
) -> tuple[float, float | None, float | None]:
    total_budget = akh_budget + ipguard_budget
    if total_budget <= 0:
        raise ValueError("The total hybrid budget must be positive")
    if len(source_labels) != total_budget or len(target_labels) != total_budget:
        raise ValueError("Label vectors must match the total hybrid budget")

    mismatches = source_labels != target_labels
    hybrid_score = mismatches.to(torch.float64).mean().item()
    akh_score = (
        mismatches[:akh_budget].to(torch.float64).mean().item()
        if akh_budget
        else None
    )
    ipguard_score = (
        mismatches[akh_budget:].to(torch.float64).mean().item()
        if ipguard_budget
        else None
    )
    return hybrid_score, akh_score, ipguard_score


def fixed_hybrid_cache_dir(
    root: Path,
    benchmark: str,
    dataset: str,
    seed: int,
    total_budget: int,
    akh_budget: int,
    ipguard_budget: int,
    source_model: str,
) -> Path:
    ratio = akh_budget / total_budget
    return (
        root
        / benchmark
        / dataset
        / f"seed={seed}"
        / f"method={FIXED_HYBRID_METHOD}"
        / f"budget={total_budget}"
        / f"akh_ratio={ratio:.2f}"
        / f"akh_budget={akh_budget},ipguard_budget={ipguard_budget}"
        / source_model
    )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_cache_manifest(path: Path, expected: dict[str, Any]) -> None:
    """Reject a cache entry when its recorded experiment identity differs."""
    if path.is_file():
        actual = json.loads(path.read_text(encoding="utf-8"))
        if actual != expected:
            raise ValueError(f"Cache manifest mismatch at {path}")
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(expected, indent=2, sort_keys=True), encoding="utf-8"
    )
    temporary_path.replace(path)
