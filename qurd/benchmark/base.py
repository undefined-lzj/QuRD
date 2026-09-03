from pathlib import Path

from abc import ABC, abstractmethod
from typing import Any, Iterable, Iterator

from torch import nn
from torch.utils.data import Dataset
from torchvision.transforms.v2 import Compose
import polars as pl

from .data import get_dataset


class Benchmark(ABC):
    base_models = {}

    input_variations = []
    output_variations = []
    model_variations = []

    def __init__(
        self,
        data_dir: Path = Path("data/"),
        models_dir: Path = Path("models/"),
        device: str = "cuda",
    ) -> None:
        self.data_dir = data_dir
        self.models_dir = models_dir
        self.device = device

    def prepare(self, datasets: Iterable[str] | str | None = None):
        """Download the datasets and test if they load properly"""

        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)

        if datasets is None:
            datasets = self.base_models.keys()
        elif isinstance(datasets, str):
            datasets = [datasets]

        for dataset in datasets:
            # Download the models
            for model in self.list_models(dataset):
                print(model)
                _ = self.torch_model(model)

            # Download the datasets
            _ = get_dataset(
                dataset,
                data_dir=self.data_dir,
                split="train",
                transform=None,
                download=True,
            )
            _ = get_dataset(
                dataset,
                data_dir=self.data_dir,
                split="test",
                transform=None,
                download=True,
            )

    def dataset(
        self,
        name: str,
        transform: Compose | None = None,
        split: str = "test",
        download: bool = False,
    ) -> Dataset:
        dataset, _ = get_dataset(
            name,
            data_dir=self.data_dir,
            split=split,
            transform=transform,
            download=download,
        )
        return dataset

    @property
    def datasets(self) -> list[str]:
        return list(self.base_models.keys())

    def pair_metadata(
        self, source_model: str, target_model: str
    ) -> dict[str, Any]:
        """Describe the ground-truth relation for a benchmark model pair."""
        if source_model == target_model:
            return {
                "is_positive": True,
                "attack_type": "same",
            }

        derived_prefix = source_model + "->"
        if target_model.startswith(derived_prefix):
            variation = target_model.removeprefix(derived_prefix).split("->")[-1]
            return {
                "is_positive": True,
                "attack_type": variation.split("(", 1)[0],
            }

        return {
            "is_positive": False,
            "attack_type": "unrelated",
        }

    @abstractmethod
    def list_models(self, dataset: str = "CIFAR10") -> Iterable[str]:
        """Lists all the models used in the benchmark, by their name"""
        ...

    @abstractmethod
    def pairs(self, dataset: str | None = None) -> Iterable[tuple[str, str]]:
        """List all the possible (source_model, target_model) pairs"""
        ...

    @abstractmethod
    def torch_model(
        self, model_name: str, from_disk: bool = False, jit: bool = False
    ) -> nn.Module:
        """Returns the torch Module given the model name.

        If `no_variation` is True, only return the base torch Module, without
        the model variations.
        """
        ...

    @abstractmethod
    def from_records(self, generated_dir: Path) -> pl.DataFrame:
        """Creates a dataframe with columns [dataset, distance, representation,
        sampler, budget, variation_name, task, source_model, target_model,
        value, unrelated mean/min/max] from records of the experiments.
        """
        raise NotImplementedError
