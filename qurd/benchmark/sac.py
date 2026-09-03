from collections import OrderedDict
from pathlib import Path
from typing import Iterable, Iterator

from torch import nn
import torch
from torch.nn.utils import prune
import torch.utils
import torchvision
import polars as pl
from timm.models import load_state_dict_from_hf, load_model_config_from_hf

from .base import Benchmark
from .utils import decompose_name, to_hf_name


class SACBenchmark(Benchmark):
    base_models = {"CIFAR10": ["train(vgg_model,CIFAR10,base)"]}

    irrelevant_models = (
        # Irrelevant models trained on CIFAR10
        [f"train(vgg13,CIFAR10,{seed})" for seed in range(5)]
        + [f"train(resnet18,CIFAR10,{5 + seed})" for seed in range(5)]
        + [f"train(densenet121,CIFAR10,{10 + seed})" for seed in range(5)]
        + [f"train(mobilenet_v2,CIFAR10,{15 + seed})" for seed in range(5)]
        # Irrelevant models trained on CIFAR10C
        + [f"train(vgg16_bn,CIFAR10C,{seed})" for seed in range(5)]
        + [f"train(resnet18,CIFAR10C,{5 + seed})" for seed in range(5)]
    )

    # fmt: off
    model_variations = {"train(vgg_model,CIFAR10,base)":([]      
        # Finetuning the model on the same dataset but different split
        + [f"finetune({seed})" for seed in range(10)]

        # Transfer the model on a different dataset
        + [f"transfer(CIFAR100,{seed})" for seed in range(10)]
        + [f"transfer(CIFAR10C,{seed})" for seed in range(10)]

        # Pruning the model to remove watermarks
        + [f"fineprune({seed})" for seed in range(10)]

        # Extracting the model from its labels
        + [f"label_extraction(vgg13,{seed})" for seed in range(5)]
        + [f"label_extraction(resnet18,{5 + seed})" for seed in range(5)]
        + [f"label_extraction(densenet121,{10 + seed})" for seed in range(5)]
        + [f"label_extraction(mobilenet_v2,{15 + seed})"for seed in range(5)]
        
        # Aversarial model extraction from labels
        + [f"adv_label_extraction(vgg13,{seed})" for seed in range(5)]
        + [f"adv_label_extraction(resnet18,{5 + seed})" for seed in range(5)]
        + [f"adv_label_extraction(densenet121,{10 + seed})" for seed in range(5)]
        + [f"adv_label_extraction(mobilenet_v2,{15 + seed})"for seed in range(5)]

        # Extracting the model from its logits
        + [f"probit_extraction(vgg13,{seed})" for seed in range(5)]
        + [f"probit_extraction(resnet18,{5 + seed})" for seed in range(5)]
        + [f"probit_extraction(densenet121,{10 + seed})" for seed in range(5)]
        + [f"probit_extraction(mobilenet_v2,{15 + seed})" for seed in range(5)]
    )}
    # fmt: on

    def pairs(self, dataset: str | None = None) -> Iterable[tuple[str, str]]:
        if dataset is None:
            dataset = "CIFAR10"

        for source_model in self.base_models[dataset]:
            for target_model in self.list_models(dataset):
                yield source_model, target_model

    def list_models(self, dataset: str = "CIFAR10") -> Iterator[str]:
        for base_model_name in self.base_models[dataset]:
            # Raw model
            yield base_model_name

            # Irrelevant models (yielded with the same name structure as a
            # base model since they are not derived from a source model).
            for variation_name in self.irrelevant_models:
                yield variation_name

            # Raw model + input variation
            for variation_name in self.input_variations:
                yield base_model_name + "->" + variation_name

            # Raw model + output variations
            for variation_name in self.output_variations:
                yield base_model_name + "->" + variation_name

            # Modification of the model weights/architecture
            if base_model_name in self.model_variations:
                for variation_name in self.model_variations[base_model_name]:
                    yield base_model_name + "->" + variation_name

    def torch_model(
        self, model_name: str, from_disk: bool = False, jit: bool = False
    ) -> nn.Module:
        # Discover the architecture, dataset and "seed" to use
        architecture, dataset, seed = "", "", ""
        for variation, params in decompose_name(model_name):
            task = variation

            if variation == "train":
                architecture, dataset, seed = params
            elif variation in ("finetune", "fineprune"):
                (seed,) = params
            elif variation == "transfer":
                dataset, seed = params
            elif variation in (
                "label_extraction",
                "adv_label_extraction",
                "probit_extraction",
            ):
                architecture, seed = params
            else:
                raise NotImplementedError(
                    f"The variation {variation} is not implemented"
                )

        # Create the model with the right architecture
        if architecture in ("vgg_model", "vgg16_bn"):
            model = torchvision.models.vgg16_bn(weights=None, num_classes=10)
            classifier = "classifier[6]"
            first_conv = "features[0]"
            num_features = 4_096

        elif architecture == "resnet18":
            model = torchvision.models.resnet18(weights=None, num_classes=10)
            classifier = "classifier[6]"
            first_conv = "fc"
            num_features = 512

        elif architecture == "vgg13":
            model = torchvision.models.vgg13(weights=None, num_classes=10)
            classifier = "classifier[6]"
            first_conv = "features[0]"
            num_features = 4_096

        elif architecture == "densenet121":
            model = torchvision.models.densenet121(weights=None, num_classes=10)
            classifier = "classifier"
            first_conv = "features.conv0"
            num_features = 1024

        elif architecture == "mobilenet_v2":
            model = torchvision.models.mobilenet_v2(weights=None, num_classes=10)
            classifier = "classifier[1]"
            first_conv = "features[0][0]"
            num_features = 1_280
        else:
            raise NotImplementedError(
                f"The architecture {architecture} is not supported"
            )

        # Return the right data normalization parameters
        if dataset == "CIFAR10":
            data_config = dict(
                mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)
            )
        elif dataset == "CIFAR10C":
            data_config = dict(
                mean=(0.4646, 0.6515, 0.5637), std=(0.1842, 0.3152, 0.2613)
            )
        elif dataset == "CIFAR100":
            data_config = dict(
                mean=(0.5071, 0.4865, 0.4409), std=(0.2673, 0.2564, 0.2762)
            )
        else:
            raise NotImplementedError(f"The dataset {dataset} is not supported")

        if from_disk:
            model = load_from_disk(
                task=task,
                dataset=str(dataset),
                model_name=model_name,
                seed=str(seed),
                architecture=model,
                models_dir=self.models_dir,
                device=self.device,
            )
            model.pretrained_cfg = {  # type: ignore
                "architecture": architecture,
                "crop_mode": "center",
                "first_conv": first_conv,
                "classifier": classifier,
                "input_size": (3, 32, 32),
                "num_classes": 10,
                "num_features": num_features,
                **data_config,
            }
        else:
            model_id = f"maurice-fp/SACBenchmark-{to_hf_name(model_name)}"
            state_dict = load_state_dict_from_hf(model_id)

            # Setup the model so that it can load weights and pruning masks
            if variation == "fineprune":
                for mask_name in filter(
                    lambda k: k.endswith("_mask"), state_dict.keys()
                ):
                    param_name, weight_name = mask_name.rsplit(".", 1)
                    weight_name = weight_name.replace("_mask", "")

                    parts = param_name.split(".")
                    prune.identity(getattr(model, parts[0])[int(parts[1])], weight_name)

            model.load_state_dict(state_dict)
            model.pretrained_cfg, _, _ = load_model_config_from_hf(model_id)

        if jit:
            model: nn.Module = torch.compile(model, mode="reduce-overhead")  # type: ignore

        return model

    def from_records(self, generated_dir: Path) -> pl.DataFrame:
        """Creates a dataframe with columns [dataset, distance, representation,
        sampler, budget, variation_name, task, source_model, target_model,
        value, unrelated mean/min/max] from records of the experiments.
        """

        tasks_no_transfer = [
            "adv_label_extraction",
            "fineprune",
            "finetune",
            "label_extraction",
            "probit_extraction",
            "same",
        ]
        source_and_dist_key = [
            "dataset",
            "split",
            "sampler",
            "representation",
            "distance",
            "budget",
            "source_model",
        ]

        records = (
            pl.scan_csv(generated_dir / "*" / "*" / "*.csv")
            .with_columns(
                # Get the variation name. Target models that have no variation
                # (i.e. that do not have a ->) are the unrelated models used for
                # each task.
                variation_name=pl.when(pl.col("source_model") != pl.col("target_model"))
                .then(
                    pl.col("target_model")
                    .str.split("->")
                    .list.slice(1)
                    .list.join("->")
                    # Remove the parameters and parentheses
                    .str.replace_all(r"\((.*?)\)", "")
                    .replace("", "unrelated")
                )
                .otherwise(pl.lit("same")),
            )
            .with_columns(
                # Repeat the unrelated models for all the tasks except the
                # CIFAR10c transfer task.
                task=pl.when(  # Unrelated models on CIFAR10
                    pl.col("variation_name") == "unrelated",
                    ~pl.col("target_model").str.contains("cifar10C"),
                )
                .then(pl.lit(tasks_no_transfer))
                # Repeat the unrelated models scpecific to the CIFAR10c transfer
                # task
                .when(  # Unrelated models on cifar10C
                    pl.col("variation_name") == "unrelated",
                    pl.col("target_model").str.contains("cifar10C"),
                )
                .then(pl.lit(["transfer"]))
                # If it is not an unrelated model, just pass the variation name
                # (and cast it into a singleton)
                .otherwise(pl.col("variation_name").cast(pl.List(pl.String)))
            )
            .explode("task")
            .collect()
        )

        # Create a column with the constants (per source) that will be used for
        # calibration
        results = records.join(
            records.group_by(source_and_dist_key)
            .agg(
                unrelated_mean=pl.col("value")
                .filter(pl.col("variation_name") == "unrelated")
                .mean(),
                unrelated_max=pl.col("value")
                .filter(pl.col("variation_name") == "unrelated")
                .max(),
                unrelated_min=pl.col("value")
                .filter(pl.col("variation_name") == "unrelated")
                .min(),
            )
            .sort(source_and_dist_key),
            on=source_and_dist_key,
        )

        return results


def load_from_disk(
    task: str,
    dataset: str,
    model_name: str,
    seed: str | int,
    architecture: nn.Module,
    models_dir: Path,
    device,
) -> nn.Module:
    """Loads the requested model from disk, assuming a setup as described in the
    original code."""

    # Resolve the weights path
    if task == "train":
        # Base (victim) model
        if "base" in model_name:
            weights_path = models_dir / "model" / "vgg_model.pth"
        # Irrelevant model model trained on CIFAR10
        elif dataset == "CIFAR10":
            weights_path = models_dir / "model" / f"clean_model_{seed}.pth"
        # Irrelevant model model trained on CIFAR10C
        elif dataset == "CIFAR10C":
            weights_path = models_dir / "finetune_model" / f"CIFAR10C_{seed}.pth"
        # Irrelevant model model trained on 10 first labels of CIFAR100
        elif dataset == "CIFAR100":
            weights_path = models_dir / "finetune_model" / f"CIFAR100{seed}.pth"
        else:
            raise NotImplementedError()

    elif task == "finetune":
        weights_path = models_dir / "finetune_10" / f"finetune{seed}.pth"

    elif task == "transfer":
        # transfer to CIFAR10C
        if dataset == "CIFAR10C":
            weights_path = (
                models_dir / "finetune_model" / f"finetune_cifar10c_{seed}.pth"
            )
        # transfer to 10 first labels of CIFAR100
        elif dataset == "CIFAR100":
            weights_path = models_dir / "finetune_model" / f"finetune_C_{seed}.pth"
        else:
            raise NotImplementedError()

    elif task == "fineprune":
        weights_path = models_dir / "Fine-Pruning" / f"prune_model_{seed}.pth"

    elif task == "label_extraction":
        weights_path = models_dir / "model" / f"student_model_1_{seed}.pth"

    elif task == "adv_label_extraction":
        weights_path = models_dir / "adv_train" / f"adv_{seed}.pth"

    elif task == "probit_extraction":
        weights_path = models_dir / "model" / f"student_model_kd_{seed}.pth"

    else:
        raise NotImplementedError(f"The task {task} is not supported")

    if task == "fineprune":
        # When creating the finepruned models, the SAC paper authors likely
        # saved the models using torch.save(). This resulted in the pickling
        # of the FeatureHook object as a link to its declaration in the
        # module __main__. Therefore, we need to add this object in the
        # __main__ module in order to unpickle the model.
        class FeatureHook:
            def __init__(self, module):
                self.hook = module.register_forward_hook(self.hook_fn)

            def hook_fn(self, module, input, output):
                self.output = output

            def close(self):
                self.hook.remove()

        main = __import__("__main__")
        main.FeatureHook = FeatureHook

        model: nn.Module = torch.load(
            weights_path, map_location=device, weights_only=False
        )

        # Remove the hooks that were added during finepruning. No other hook
        # is present here so we can delete them all.
        def delete_hooks(module: nn.Module):
            for child in module.children():
                child._forward_hooks = OrderedDict()

        delete_hooks(model)

    else:
        model = architecture
        model.load_state_dict(
            torch.load(weights_path, map_location=device, weights_only=True)
        )

    return model
