"""Read-only preflight for the first Fixed-Hybrid experiment."""

from __future__ import annotations

import csv
import hashlib
import pickle
from pathlib import Path

import torch


SOURCE = "train(vgg_model,CIFAR10,base)"
AKH_QUERIES = Path(
    "generated/SACBenchmark/CIFAR10/"
    "RandomNegativeQueries(augment=True,subsample=None)-20/"
) / f"{SOURCE}.pickle"
IPGUARD_QUERIES = Path(
    "generated/SACBenchmark/CIFAR10/BoundaryQueries(k=10)-20/"
) / f"{SOURCE}.pickle"
BASELINES = [
    Path(
        "generated/results/phase2b2_sacbench_cifar10_s123456789/"
        "phase2b2_sacbench_cifar10_akh_b20_s123456789.csv"
    ),
    Path(
        "generated/results/phase2b2_sacbench_cifar10_s123456789/"
        "phase2b2_sacbench_cifar10_ipguard_b20_s123456789.csv"
    ),
]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_query_pool(path: Path) -> torch.Tensor:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("rb") as handle:
        pool = pickle.load(handle)
    if not isinstance(pool, torch.Tensor) or len(pool) != 20:
        raise ValueError(f"{path} must contain exactly 20 tensor queries")
    return pool


def load_metadata(path: Path) -> dict[tuple[str, str], tuple[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 131:
        raise ValueError(f"{path} must contain exactly 131 rows")
    metadata = {
        (row["source_model"], row["target_model"]): (
            row["is_positive"].lower(),
            row["attack_type"],
        )
        for row in rows
    }
    if len(metadata) != 131:
        raise ValueError(f"{path} contains duplicate model pairs")
    if sum(value[0] == "true" for value in metadata.values()) != 101:
        raise ValueError(f"{path} must contain 101 positive model pairs")
    return metadata


def main() -> None:
    load_query_pool(AKH_QUERIES)
    load_query_pool(IPGUARD_QUERIES)
    akh_metadata = load_metadata(BASELINES[0])
    ipguard_metadata = load_metadata(BASELINES[1])
    if akh_metadata != ipguard_metadata:
        raise ValueError("AKH and IPGuard budget=20 baselines use different metadata")

    print("Fixed-Hybrid preflight passed")
    print(f"model_pairs=131 positive=101 negative=30")
    print(f"akh_query_pool_sha256={file_sha256(AKH_QUERIES)}")
    print(f"ipguard_query_pool_sha256={file_sha256(IPGUARD_QUERIES)}")


if __name__ == "__main__":
    main()
