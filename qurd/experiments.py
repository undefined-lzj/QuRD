from pathlib import Path
import csv
import pickle
from time import perf_counter
from typing import Any, Callable, Iterable, TypeVar
import json

from timm.data import create_transform, resolve_data_config
import torch
from torch import nn
from torchmetrics import Accuracy
from torch.utils.data import DataLoader
from tqdm import tqdm

from .benchmark.base import Benchmark
from .fingerprint.base import OutputRepresentation, QueriesSampler
from .fingerprint.utils import split_transform
from .fixed_hybrid import (
    FIXED_HYBRID_METHOD,
    FixedHybridScoresCsv,
    ensure_cache_manifest,
    file_sha256,
    fixed_hybrid_cache_dir,
    make_fixed_hybrid_queries,
    score_fixed_hybrid_labels,
)


SCORE_COLUMNS = [
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
SCORE_KEY_COLUMNS = [
    "dataset",
    "method",
    "budget",
    "seed",
    "source_model",
    "target_model",
]


class ScoresCsv:
    """Persist model-pair scores to a deduplicated CSV file."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._records: dict[tuple[str, ...], dict[str, Any]] = {}

        if path.is_file():
            with path.open(newline="", encoding="utf-8") as file:
                for record in csv.DictReader(file):
                    normalized = self._normalize(record)
                    self._records[self._key(normalized)] = normalized

    def upsert(self, record: dict[str, Any]) -> None:
        """Insert or replace one experiment result and flush it atomically."""
        normalized = self._normalize(record)
        self._records[self._key(normalized)] = normalized
        self._flush()

    @staticmethod
    def _normalize(record: dict[str, Any]) -> dict[str, Any]:
        source_model = str(record["source_model"])
        target_model = str(record["target_model"])
        seed = record.get("seed", "")
        seed = "" if seed in (None, "") else int(seed)

        default_positive = source_model == target_model or target_model.startswith(
            source_model + "->"
        )
        raw_positive = record.get("is_positive", "")
        if raw_positive in (None, ""):
            # Backward compatibility for CSV files written before the
            # redundant text label column was removed.
            legacy_label = str(record.get("label", "")).strip().lower()
            if legacy_label in ("positive", "negative"):
                raw_positive = legacy_label == "positive"
            else:
                raw_positive = default_positive
        if isinstance(raw_positive, str):
            is_positive = raw_positive.lower() in ("1", "true", "yes")
        else:
            is_positive = bool(raw_positive)

        if source_model == target_model:
            default_attack = "same"
        elif target_model.startswith(source_model + "->"):
            variation = target_model.removeprefix(source_model + "->").split("->")[-1]
            default_attack = variation.split("(", 1)[0]
        else:
            default_attack = "unrelated"

        return {
            "method": str(record["method"]),
            "budget": int(record["budget"]),
            "seed": seed,
            "source_model": source_model,
            "target_model": target_model,
            "is_positive": is_positive,
            "attack_type": str(record.get("attack_type", default_attack)),
            "score": float(record["score"]),
            "dataset": str(record["dataset"]),
        }

    @staticmethod
    def _key(record: dict[str, Any]) -> tuple[str, ...]:
        return tuple(str(record[column]) for column in SCORE_KEY_COLUMNS)

    def _flush(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = self.path.with_suffix(self.path.suffix + ".tmp")

        with temporary_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=SCORE_COLUMNS)
            writer.writeheader()
            writer.writerows(self._records.values())

        temporary_path.replace(self.path)


class Experiment:
    """Run a fingerprint evaluation benchmark

    - Cache the queries and representations
    - Manage model loading
    - Compute the desired metrics
    """

    def __init__(
        self,
        benchmark: Benchmark,
        dir: Path,
        batch_size: int,
        device: str,
        seed: int | None = None,
    ) -> None:
        self.benchmark = benchmark
        self.batch_size = batch_size
        self.device = device
        self.dir = dir
        self.seed = seed

    def eval_models(
        self, datasets: Iterable[str] | str | None = None, jit: bool = False
    ):
        """Compute the accuracy of the models in the benchmark.

        You can specify a subset of datasets with the `datasets` argument
        """
        if datasets is None:
            datasets = self.benchmark.base_models.keys()
        elif isinstance(datasets, str):
            datasets = [datasets]

        datasets_accuracies = {}

        for dataset_name in datasets:

            # Prepare where the results will be stored
            save_path = (
                self.dir
                / self.benchmark.__class__.__name__
                / dataset_name
                / "accuracy.json"
            )
            save_path.parent.mkdir(parents=True, exist_ok=True)
            if save_path.is_file():
                models_accuracy = json.loads(save_path.read_text())

            else:
                models_accuracy = {}

            for model_name in self.benchmark.list_models(dataset_name):
                start = perf_counter()

                # Check that the accuracy was not already computed
                # if model_name in models_accuracy:
                #     continue

                print(model_name, end="... ", flush=True)

                model = self.benchmark.torch_model(model_name, jit=jit)

                # Get the dataset with the right transform
                transform = create_transform(
                    **resolve_data_config(model.pretrained_cfg)
                )

                # TODO: speedup dataset loading by loading dataset once, and
                # just changing the transform every time
                dataset = self.benchmark.dataset(dataset_name, transform=transform)
                test_loader = DataLoader(
                    dataset,
                    batch_size=self.batch_size,
                    num_workers=4,
                    pin_memory=True,
                )

                device = self.device
                if "quantize" in model_name:
                    device = "cpu"

                # Send model to device
                model.eval()
                model.to(device)

                # Prepare the metric
                accuracy = Accuracy(
                    task="multiclass",
                    num_classes=model.pretrained_cfg["num_classes"],
                    top_k=1,
                ).to(device)

                # Run the inference
                for images, labels in test_loader:
                    images = images.to(device)
                    preds: torch.Tensor = model(images).argmax(dim=-1)

                    acc = accuracy(preds, labels.to(device))

                    labels.cpu()
                    preds.cpu()
                    images.cpu()

                # Unload model and save everything in case of crash
                model.cpu()
                models_accuracy[model_name] = accuracy.compute().cpu().item()
                save_path.write_text(json.dumps(models_accuracy))

                print(
                    f"{perf_counter() - start:.3f} s Top1 = {models_accuracy[model_name]:.2f}"
                )

            datasets_accuracies[dataset_name] = models_accuracy

        return datasets_accuracies

    def scores(
        self,
        fingerprints: dict[str, tuple[QueriesSampler, OutputRepresentation]],
        budget: int,
        scores_path: Path | None = None,
    ):
        """Run the fingerprints on the benchmark and compute the fingerprinting
        scores.

        - It caches the queries that are sampled by the `QuerySampler` of each
          fingerprint. The cache is shared accross fingerprints. That is, if two
          fingerprints have the same queries sampler (with the same parameters)
          then they will have the same queries. This allows to make sure that
          two competing fingerprints that share the same sampler also share the
          randomness.
        - It caches the representations similarly.
        """

        scores: list[dict[str, Any]] = []
        scores_csv = ScoresCsv(scores_path or self.dir / "scores.csv")
        current_source_name: str | None = None
        current_source_model: nn.Module | None = None

        for dataset_name in self.benchmark.base_models:
            dataset = self.benchmark.dataset(dataset_name)

            for fingerprint, (
                sampler,
                representation,
                distance,
            ) in fingerprints.items():
                n_pairs = len(list(self.benchmark.pairs(dataset_name)))
                progress = tqdm(
                    self.benchmark.pairs(dataset_name),
                    total=n_pairs,
                    desc=f"{dataset_name}: {fingerprint}",
                    position=1,
                )

                for source_name, target_name in progress:
                    progress.display(f"{source_name} vs {target_name}", pos=2)
                    # print(source_name, target_name)

                    if source_name != current_source_name:
                        if current_source_model is not None:
                            current_source_model.cpu()
                        current_source_model = self.benchmark.torch_model(source_name)
                        current_source_name = source_name

                    source_model = current_source_model
                    target_model = (
                        source_model
                        if target_name == source_name
                        else self.benchmark.torch_model(target_name)
                    )
                    source_transform = create_transform(
                        **resolve_data_config(source_model.pretrained_cfg)
                    )
                    target_transform = create_transform(
                        **resolve_data_config(target_model.pretrained_cfg)
                    )

                    # adapt the device for quantized models
                    source_device = "cpu" if "quantize" in source_name else self.device
                    target_device = "cpu" if "quantize" in target_name else self.device

                    source_model = source_model.to(source_device)
                    target_model = target_model.to(target_device)
                    # print(f"{source_device = }, {target_device = }")

                    # Query samplers must receive image-domain tensors. Keep
                    # resizing/cropping on the dataset, but leave normalization
                    # to the sampler/model evaluation path so it is applied once.
                    query_transform, _ = split_transform(source_transform)
                    dataset.transform = query_transform

                    # Compute the queries
                    queries_path: Path = (
                        self.dir
                        / self.benchmark.__class__.__name__
                        / dataset_name
                        / (str(sampler) + "-" + str(budget))
                        / (source_name + ".pickle")
                    )
                    sampler.device = source_device
                    queries = _load_or_compute(
                        lambda: sampler.sample(
                            dataset=dataset,
                            budget=budget,
                            source_model=source_model,
                            source_transform=source_transform,
                        ),
                        queries_path,
                    )
                    # # Does not work with some querysamplers
                    # torchvision.utils.save_image(
                    #     torch.cat(queries), queries_path.with_suffix(".png"), nrow=5
                    # )

                    # Compute the source representation
                    source_repr_path: Path = (
                        self.dir
                        / self.benchmark.__class__.__name__
                        / dataset_name
                        / (str(sampler) + "-" + str(budget))
                        / str(representation)
                        / source_name
                        / "source.pickle"
                    )
                    representation.device = source_device
                    source_repr = _load_or_compute(
                        lambda: representation(
                            queries=queries,
                            model=source_model,
                            transform=source_transform,
                        ),
                        source_repr_path,
                    )

                    # Compute the target representation
                    target_repr_path: Path = (
                        self.dir
                        / self.benchmark.__class__.__name__
                        / dataset_name
                        / (str(sampler) + "-" + str(budget))
                        / str(representation)
                        / source_name
                        / (target_name + ".pickle")
                    )
                    representation.device = target_device
                    target_repr = _load_or_compute(
                        lambda: representation(
                            queries=queries,
                            model=target_model,
                            transform=target_transform,
                        ),
                        target_repr_path,
                    )

                    source_repr.cpu()
                    target_repr.cpu()

                    # Compute the distance
                    score = distance(source_repr, target_repr)

                    # Persist each pair immediately so an interrupted experiment
                    # keeps all results completed up to that point.
                    pair_metadata = self.benchmark.pair_metadata(
                        source_name, target_name
                    )
                    record = dict(
                        dataset=dataset_name,
                        method=fingerprint,
                        budget=budget,
                        seed=self.seed,
                        source_model=source_name,
                        target_model=target_name,
                        score=score,
                        **pair_metadata,
                    )
                    scores_csv.upsert(record)
                    scores.append(record)

                    # Unload models from the GPU
                    source_model.cpu()
                    if target_model is not source_model:
                        target_model.cpu()
                        del target_model

        return scores

    def fixed_hybrid_scores(
        self,
        budget: int,
        akh_budgets: Iterable[int],
        scores_path: Path,
        baseline_cache_dir: Path,
    ) -> list[dict[str, Any]]:
        """Evaluate fixed AKH/IPGuard query mixtures using hard-label distance.

        The component queries are fixed prefixes of the existing AKH and
        IPGuard baseline query pools. Query construction uses no target-model
        information. Every active cache path and manifest records the seed,
        method, total budget, and mixture ratio.
        """
        from .fingerprint.queries import BoundaryQueries, RandomNegativeQueries
        from .fingerprint.representation import HardLabels

        akh_budgets = tuple(akh_budgets)
        if budget <= 0:
            raise ValueError("budget must be positive")
        if any(value < 0 or value > budget for value in akh_budgets):
            raise ValueError("Each AKH budget must be between zero and budget")
        if len(set(akh_budgets)) != len(akh_budgets):
            raise ValueError("AKH budgets must be unique")
        if self.seed is None:
            raise ValueError("Fixed-Hybrid requires an explicit seed")

        writer = FixedHybridScoresCsv(scores_path)
        records: list[dict[str, Any]] = []
        representation = HardLabels(batch_size=self.batch_size, device=self.device)
        akh_sampler = RandomNegativeQueries(
            augment=True, device=self.device, batch_size=self.batch_size
        )
        ipguard_sampler = BoundaryQueries(
            batch_size=self.batch_size, device=self.device
        )

        for dataset_name in self.benchmark.base_models:
            dataset = self.benchmark.dataset(dataset_name)
            pairs = list(self.benchmark.pairs(dataset_name))
            progress = tqdm(
                pairs,
                total=len(pairs),
                desc=f"{dataset_name}: {FIXED_HYBRID_METHOD}",
                position=1,
            )

            current_source_name: str | None = None
            current_source_model: nn.Module | None = None
            query_sets: dict[int, tuple[torch.Tensor, torch.Tensor, Path]] = {}

            for source_name, target_name in progress:
                progress.display(f"{source_name} vs {target_name}", pos=2)

                if source_name != current_source_name:
                    if current_source_model is not None:
                        current_source_model.cpu()
                    current_source_model = self.benchmark.torch_model(source_name)
                    current_source_name = source_name
                    source_device = (
                        "cpu" if "quantize" in source_name else self.device
                    )
                    current_source_model = current_source_model.to(source_device)
                    source_transform = create_transform(
                        **resolve_data_config(current_source_model.pretrained_cfg)
                    )
                    query_transform, _ = split_transform(source_transform)
                    dataset.transform = query_transform

                    akh_pool_path = (
                        baseline_cache_dir
                        / self.benchmark.__class__.__name__
                        / dataset_name
                        / f"{akh_sampler}-{budget}"
                        / f"{source_name}.pickle"
                    )
                    ipguard_pool_path = (
                        baseline_cache_dir
                        / self.benchmark.__class__.__name__
                        / dataset_name
                        / f"{ipguard_sampler}-{budget}"
                        / f"{source_name}.pickle"
                    )
                    if not akh_pool_path.is_file() or not ipguard_pool_path.is_file():
                        raise FileNotFoundError(
                            "Fixed-Hybrid requires the completed AKH and IPGuard "
                            f"budget={budget} baseline query caches for {source_name}"
                        )
                    akh_pool: torch.Tensor = _load(akh_pool_path)
                    ipguard_pool: torch.Tensor = _load(ipguard_pool_path)
                    if len(akh_pool) != budget or len(ipguard_pool) != budget:
                        raise ValueError(
                            "Baseline query pools must each contain exactly "
                            f"{budget} queries"
                        )

                    akh_pool_hash = file_sha256(akh_pool_path)
                    ipguard_pool_hash = file_sha256(ipguard_pool_path)
                    query_sets = {}
                    for akh_budget in akh_budgets:
                        ipguard_budget = budget - akh_budget
                        cache_dir = fixed_hybrid_cache_dir(
                            self.dir,
                            self.benchmark.__class__.__name__,
                            dataset_name,
                            self.seed,
                            budget,
                            akh_budget,
                            ipguard_budget,
                            source_name,
                        )
                        query_path = cache_dir / "queries.pickle"
                        manifest_path = cache_dir / "manifest.json"
                        if query_path.is_file() and not manifest_path.is_file():
                            raise ValueError(
                                f"Refusing unidentifiable cache without manifest: {query_path}"
                            )
                        manifest = {
                            "method": FIXED_HYBRID_METHOD,
                            "benchmark": self.benchmark.__class__.__name__,
                            "dataset": dataset_name,
                            "seed": self.seed,
                            "budget": budget,
                            "akh_budget": akh_budget,
                            "ipguard_budget": ipguard_budget,
                            "akh_ratio": akh_budget / budget,
                            "source_model": source_name,
                            "query_selection": "fixed_prefix_from_completed_baseline_pools",
                            "target_model_used_for_queries": False,
                            "akh_pool_path": str(akh_pool_path.resolve()),
                            "akh_pool_sha256": akh_pool_hash,
                            "ipguard_pool_path": str(ipguard_pool_path.resolve()),
                            "ipguard_pool_sha256": ipguard_pool_hash,
                        }
                        ensure_cache_manifest(manifest_path, manifest)
                        expected_queries = make_fixed_hybrid_queries(
                            akh_pool,
                            ipguard_pool,
                            akh_budget,
                            ipguard_budget,
                        )
                        queries = _load_or_compute(
                            lambda q=expected_queries: q,
                            query_path,
                        )
                        if len(queries) != budget or not torch.equal(
                            queries, expected_queries
                        ):
                            raise ValueError(
                                "Hybrid query cache differs from its fixed baseline prefixes"
                            )
                        representation.device = source_device
                        source_labels_path = (
                            cache_dir / str(representation) / "source.pickle"
                        )
                        source_labels = _load_or_compute(
                            lambda q=queries: representation(
                                queries=q,
                                model=current_source_model,
                                transform=source_transform,
                            ),
                            source_labels_path,
                        )
                        query_sets[akh_budget] = (
                            queries,
                            source_labels,
                            cache_dir,
                        )

                source_model = current_source_model
                if source_model is None:
                    raise RuntimeError("Source model was not initialized")
                source_transform = create_transform(
                    **resolve_data_config(source_model.pretrained_cfg)
                )
                target_model = (
                    source_model
                    if target_name == source_name
                    else self.benchmark.torch_model(target_name)
                )
                target_device = "cpu" if "quantize" in target_name else self.device
                target_model = target_model.to(target_device)
                target_transform = create_transform(
                    **resolve_data_config(target_model.pretrained_cfg)
                )

                for akh_budget in akh_budgets:
                    ipguard_budget = budget - akh_budget
                    queries, source_labels, cache_dir = query_sets[akh_budget]
                    if target_model is source_model:
                        target_labels = source_labels
                    else:
                        representation.device = target_device
                        target_labels_path = (
                            cache_dir
                            / str(representation)
                            / "targets"
                            / f"{target_name}.pickle"
                        )
                        target_labels = _load_or_compute(
                            lambda q=queries: representation(
                                queries=q,
                                model=target_model,
                                transform=target_transform,
                            ),
                            target_labels_path,
                        )

                    hybrid_score, akh_score, ipguard_score = (
                        score_fixed_hybrid_labels(
                            source_labels,
                            target_labels,
                            akh_budget,
                            ipguard_budget,
                        )
                    )
                    pair_metadata = self.benchmark.pair_metadata(
                        source_name, target_name
                    )
                    record = {
                        "method": FIXED_HYBRID_METHOD,
                        "budget": budget,
                        "seed": self.seed,
                        "source_model": source_name,
                        "target_model": target_name,
                        "score": hybrid_score,
                        "dataset": dataset_name,
                        "pair_label": int(pair_metadata["is_positive"]),
                        "attack_type": pair_metadata["attack_type"],
                        "akh_budget": akh_budget,
                        "ipguard_budget": ipguard_budget,
                        "akh_ratio": akh_budget / budget,
                        "akh_score": akh_score,
                        "ipguard_score": ipguard_score,
                        "hybrid_score": hybrid_score,
                    }
                    writer.upsert(record)
                    records.append(record)

                if target_model is not source_model:
                    target_model.cpu()
                    del target_model

            if current_source_model is not None:
                current_source_model.cpu()

        return records


T = TypeVar("T")


def _cache(obj: T, to: Path):
    to.parent.mkdir(parents=True, exist_ok=True)

    with open(to, "wb") as file:
        pickle.dump(obj, file)


def _load(from_path: Path) -> T:
    with open(from_path, "rb") as file:
        return pickle.load(file)


def _load_or_compute(func: Callable[[], T], path: Path) -> T:
    if path.is_file():
        return _load(path)

    result = func()
    _cache(result, path)

    return result
