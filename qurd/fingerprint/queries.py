from math import ceil
from pathlib import Path
from typing import Any, Optional

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2, Compose, Normalize
from torchvision.transforms.v2 import functional as VF

from .base import QueriesSampler
from .modeldiff import find_adversarial
from .zlime import subsample
from .utils import batch_predict, sample_batch, split_transform


class RandomQueries(QueriesSampler):
    """Sample queries uniformly in the provided dataset.

    If a `subsample` parameter is provided, first uniformly sample budget //
    subsample seed inputs. Then create subsample images from each seed image
    using LIME image augmentation.
    """

    def __init__(self, subsample: Optional[int] = None):
        self.subsample = subsample

    def sample(
        self, dataset: Dataset, budget: int, **kwargs
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        if self.subsample:
            assert (
                budget % self.subsample == 0
            ), f"The budget value {budget} must be divisible by the subsampling ratio {self.subsample}"

        if self.subsample:
            assert (
                budget % self.subsample == 0
            ), "The budget must be divisible by the subsampling factor"

            seed_images, _ = sample_batch(dataset, n=budget // self.subsample)
            images, lime_features = subsample(seed_images.cpu().numpy(), self.subsample)

            return images, lime_features

        else:
            images, _ = sample_batch(dataset, n=budget)

        return images

    def from_file(
        self, path: Path, flatten: bool = False, return_lime_feature: bool = False
    ) -> Any:
        images = super().from_file(path)

        if self.subsample:
            images, lime_features = images

        if flatten:
            # Flatten the images to get a tensor of dim [N, C, W, H]
            images = torch.flatten(images, start_dim=0, end_dim=-4)

        if return_lime_feature:
            return images, lime_features
        else:
            return images


class RandomNegativeQueries(QueriesSampler):
    """Sample queries uniformly in among queries that are not well classified by
    the provided `source_model`.

    - If a `subsample` parameter is provided, first uniformly sample `budget //
    subsample` seed inputs among inputs that are wrongly classified by the
    `source_model`. Then create subsample images from each seed image using LIME
    image aumentation.

    - If the `augment` parameter is true, first uniformly sample `budget //
    subsample` seed inputs among inputs that are wrongly classified by the
    `source_model`. Then, if the number of found images is lower than the
    budget, generate new images using CutMix and flip data-augmentation
    techniques.

    - `subsample` and `augment` cannot be both true at the same time.

    The inference of the source model is run on the specified `device` with the
    specified `batch_size`
    """

    def __init__(
        self,
        subsample: Optional[int] = None,
        augment: bool = False,
        device: str = "cpu",
        batch_size: int = 64,
    ):
        assert not (
            augment and subsample
        ), "Subsample and augment cannot be both activated at the same time"

        self.subsample = subsample
        self.augment = augment
        self.device = device
        self.batch_size = batch_size

    def sample(
        self,
        dataset: Dataset,
        budget: int,
        source_model: nn.Module,
        source_transform: v2.Transform,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        if self.subsample:
            assert (
                budget % self.subsample == 0
            ), f"The budget value {budget} must be divisible by the subsampling ratio {self.subsample}"

        # Gather queries with negative labels up to the budget
        candidate_queries, candidate_labels = find_negatives(
            dataset,
            transform=source_transform,
            model=source_model,
            limit=budget,
            batch_size=self.batch_size,
            device=self.device,
        )

        # Number of wrongly classified images that were found
        n_candidates = candidate_queries.shape[0]

        # Augment the querie with cut mix
        if self.augment and n_candidates <= budget:
            # The total number of generated images is n_candidates * 3: CutMix +
            # CutMix_flipped_h + CutMix_flipped_v
            n_cutmix = ceil((budget - n_candidates) / (n_candidates * 3))

            cutmix = v2.CutMix(
                alpha=1.0, num_classes=source_model.pretrained_cfg["num_classes"]
            )
            subsampled_queries = [candidate_queries]

            for _ in range(n_cutmix):
                cutmix_queries, _ = cutmix(candidate_queries, candidate_labels)
                flipped_h = VF.horizontal_flip(cutmix_queries)
                flipped_v = VF.vertical_flip(cutmix_queries)

                subsampled_queries.extend([cutmix_queries, flipped_h, flipped_v])

            candidate_queries = torch.cat(subsampled_queries, dim=0)[:budget]

        # Augment queries with LIME subsampling
        if self.subsample:
            idx = torch.randperm(candidate_queries.shape[0])[: budget // self.subsample]
            seed_images = candidate_queries[idx]
            images, lime_features = subsample(seed_images.cpu().numpy(), self.subsample)

            return images, lime_features

        # No query augmentation
        else:
            assert (
                candidate_queries.shape[0] >= budget
            ), f"Not enough candidate images {candidate_queries.shape[0]} to fill the requested budget {budget}"

            images = candidate_queries[:budget]

        return images.cpu()

    def from_file(
        self, path: Path, flatten: bool = False, return_lime_feature: bool = False
    ) -> Any:
        images = super().from_file(path)

        if self.subsample:
            images, lime_features = images

        if flatten:
            # Flatten the images to get a tensor of dim [N, C, W, H]
            images = torch.flatten(images, start_dim=0, end_dim=-4)

        if return_lime_feature:
            return images, lime_features
        else:
            return images


class AdversarialQueries(RandomQueries):
    def __init__(
        self,
        subsample: int | None = None,
        batch_size: int = 64,
        device: str = "cpu",
    ):
        super().__init__(subsample)
        self.batch_size = batch_size
        self.device = device

    def sample(
        self,
        dataset: Dataset,
        budget: int,
        source_model: nn.Module,
        source_transform: v2.Transform,
    ):
        seeds = super().sample(dataset, budget // 2)

        if prepocessing := get_normalize_params(source_transform):
            self.preprocessing = prepocessing
        else:
            raise ValueError("No Normalization was given")

        adversarial = find_adversarial(
            source_model,
            images=seeds,  # type: ignore
            preprocessing=self.preprocessing,
            batch_size=self.batch_size,
            device=self.device,
        ).cpu()

        return seeds, adversarial

    def from_file(self, path: Path, flatten: bool = False) -> Any:
        seeds, adversarial = super().from_file(path)

        if flatten:
            return torch.cat((seeds, adversarial), dim=0)

        return seeds, adversarial


class BoundaryQueries(RandomQueries):
    def __init__(
        self,
        k: float = 10,
        batch_size: int = 64,
        device: str = "cpu",
    ):
        super().__init__(subsample=None)
        self.k = k
        self.batch_size = batch_size
        self.device = device

    def sample(
        self,
        dataset: Dataset,
        budget: int,
        source_model: nn.Module | None = None,
        source_transform: v2.Transform | None = None,
    ):
        if prepocessing := get_normalize_params(source_transform):
            self.preprocessing = prepocessing
        else:
            raise ValueError("No Normalization was given")

        seeds: torch.Tensor = super().sample(dataset, budget)  # type: ignore

        logits = batch_predict(
            source_model,
            seeds,
            source_transform,
            self.batch_size,
            self.device,
        )
        n_classes = logits.shape[1]

        # The labels predicted by the source model (the i in equation (5))
        predicted_labels = logits.argmax(-1)
        # The target labels (j in the equation (5))
        target_labels = torch.randint_like(predicted_labels, low=0, high=n_classes)

        def boundary_loss(logits: torch.Tensor, source_labels, target_labels, k):
            (n_classes,) = logits.shape
            source_encoding = (
                F.one_hot(source_labels, n_classes).float().to(logits.device)
            )
            target_encoding = (
                F.one_hot(target_labels, n_classes).float().to(logits.device)
            )

            Z_i = torch.dot(source_encoding, logits)
            Z_j = torch.dot(target_encoding, logits)
            max_Z_t = torch.max(
                (1 - target_encoding) * (1 - source_encoding) * logits, -1
            ).values

            return F.relu(Z_i - Z_j + k) + F.relu(max_Z_t - Z_i)

        images = seeds.clone().detach()
        generated_images = []

        source_model.to(self.device).eval()
        for p in source_model.parameters():
            p.requires_grad = False

        for image, predicted_label, target_label in zip(
            images, predicted_labels, target_labels
        ):
            image = image.clone().detach().to(self.device)
            image.requires_grad = True
            optimizer = torch.optim.Adam([image], lr=0.01)

            # Max number of iterations is set to 1000 in IPGuard paper
            for i in range(1_000):
                pred = source_model(source_transform(image).unsqueeze(0)).squeeze()
                cost = boundary_loss(pred, predicted_label, target_label, self.k)

                cost.backward()
                optimizer.step()
                optimizer.zero_grad()

                if pred.argmax(-1) == target_label:
                    break

            generated_images.append(image.detach().cpu().unsqueeze(0))

        generated_images = torch.cat(generated_images)

        return generated_images

    def from_file(self, path: Path, flatten: bool = False) -> Any:
        images = super().from_file(path)

        return images


class AdversarialNegativeQueries(RandomNegativeQueries):
    def __init__(self, *args, source_transform: v2.Transform, **kwargs):
        super().__init__(*args, source_transform=source_transform, **kwargs)

        if prepocessing := get_normalize_params(source_transform):
            self.preprocessing = prepocessing
        else:
            raise ValueError("No Normalization was given")

    def sample(self, dataset: Dataset, budget: int):
        seeds = super().sample(dataset, budget=budget // 2)

        if self.subsample:
            seeds, _ = seeds

        adversarial = find_adversarial(
            self.source_model,
            images=seeds,  # type: ignore
            preprocessing=self.preprocessing,
            batch_size=self.batch_size,
            device=self.device,
        ).cpu()

        return seeds, adversarial

    def from_file(self, path: Path, flatten: bool = False) -> Any:
        seeds, adversarial = super().from_file(path)

        if flatten:
            return torch.cat((seeds, adversarial), dim=0)

        return seeds, adversarial


def get_normalize_params(
    transform: v2.Transform,
) -> dict[str, list[float] | float] | None:
    """Get the parameters of the Normalization in the transform (which can be a
    composition of other transforms and a normalization) to be used by foolbox"""

    if isinstance(transform, (v2.Compose, Compose)):
        for step in transform.transforms:
            if isinstance(step, (v2.Normalize, Normalize)):
                return {"mean": step.mean, "std": step.std, "axis": -3}

        return None

    elif isinstance(transform, (v2.Normalize, Normalize)):
        return {"mean": transform.mean, "std": transform.std, "axis": -3}

    return None


def find_negatives(
    dataset: Dataset,
    model: nn.Module,
    limit: int,
    batch_size: int,
    device: str,
    transform: v2.Transform | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the images (and corresponding labels) in the `dataset` that are
    wrongly classified by the provided `model`"""

    # assert (
    #     dataset.transform is None
    # ), "The dataset must have no transform as transforms are handeld by the query sampler"

    transform, normalize = split_transform(transform)
    dataset.transform = transform

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model.eval()
    model.to(device)

    # Get the images that are ill-classified by the source model
    negative_queries = []
    negative_labels = []
    total_negatives = 0

    for images, labels in dataloader:
        # print(images)
        images = images.to(device)

        with torch.no_grad():
            # Make the prediction on the transformed image
            pred_labels = model(normalize(images)).argmax(-1).cpu()
        images.cpu()

        negatives = pred_labels != labels

        negative_queries.append(images[negatives])
        negative_labels.append(labels[negatives])

        # Stop the search for negatives if we have found enough
        total_negatives += negatives.sum().item()
        if total_negatives > limit:
            break

    dataset.transform = None

    # Gather all queries (and labels)
    return torch.cat(negative_queries, dim=0), torch.cat(negative_labels, dim=0)
