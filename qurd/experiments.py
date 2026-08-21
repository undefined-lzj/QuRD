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


SCORE_COLUMNS = [
    "method",
    "budget",
    "source_model",
    "target_model",
    "score",
    "dataset",
]
SCORE_KEY_COLUMNS = [
    "dataset",
    "method",
    "budget",
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
        return {
            "method": str(record["method"]),
            "budget": int(record["budget"]),
            "source_model": str(record["source_model"]),
            "target_model": str(record["target_model"]),
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
        self, benchmark: Benchmark, dir: Path, batch_size: int, device: str
    ) -> None:
        self.benchmark = benchmark
        self.batch_size = batch_size
        self.device = device
        self.dir = dir

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
        scores_csv = ScoresCsv(self.dir / "scores.csv")
        models = {}

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

                    source_model: nn.Module = models.setdefault(
                        source_name, self.benchmark.torch_model(source_name)
                    )
                    target_model: nn.Module = models.setdefault(
                        target_name, self.benchmark.torch_model(target_name)
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

                    # Set the transform of the dataset to be that of the
                    # source_model
                    #
                    # TODO: normalize the source/target model names before saving
                    dataset.transform = source_transform

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
                    record = dict(
                        dataset=dataset_name,
                        method=fingerprint,
                        budget=budget,
                        source_model=source_name,
                        target_model=target_name,
                        score=score,
                    )
                    scores_csv.upsert(record)
                    scores.append(record)

                    # Unload models from the GPU
                    source_model.cpu()
                    target_model.cpu()

        return scores


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
