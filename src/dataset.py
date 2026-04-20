from __future__ import annotations

import gzip
import shutil
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms

from utils import PROJECT_ROOT, ensure_dir

KAGGLE_DATASET = "crawford/emnist"
TORCHVISION_ROOT = PROJECT_ROOT / "data" / "raw"
EMNIST_RAW_DIR = TORCHVISION_ROOT / "EMNIST" / "raw"
BALANCED_FILES = [
    "emnist-balanced-train-images-idx3-ubyte",
    "emnist-balanced-train-labels-idx1-ubyte",
    "emnist-balanced-test-images-idx3-ubyte",
    "emnist-balanced-test-labels-idx1-ubyte",
]


class EMNISTOrientationCorrection:
    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        tensor = torch.rot90(tensor, -1, dims=[1, 2])
        return torch.flip(tensor, dims=[1])


class GaussianNoise:
    def __init__(self, std: float = 0.0) -> None:
        self.std = float(std)

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.std <= 0:
            return tensor
        noise = torch.randn_like(tensor) * self.std
        return torch.clamp(tensor + noise, 0.0, 1.0)


def resolve_augmentation_config(augmentation_config: dict | None) -> dict:
    if augmentation_config is None:
        augmentation_config = {}
    if not isinstance(augmentation_config, dict):
        raise ValueError("augmentation config must be a mapping when provided.")

    translate = augmentation_config.get("translate", 0.0)
    if isinstance(translate, (int, float)):
        translate = (float(translate), float(translate))
    else:
        translate = tuple(float(value) for value in translate)
        if len(translate) != 2:
            raise ValueError("augmentation.translate must contain exactly two values.")

    return {
        "enabled": bool(augmentation_config.get("enabled", False)),
        "rotation_deg": float(augmentation_config.get("rotation_deg", 0.0)),
        "translate": translate,
        "gaussian_noise_std": float(augmentation_config.get("gaussian_noise_std", 0.0)),
    }


def build_transform(train: bool = False, augmentation_config: dict | None = None) -> transforms.Compose:
    resolved_augmentation = resolve_augmentation_config(augmentation_config)

    transform_steps: list = [
        transforms.ToTensor(),
        EMNISTOrientationCorrection(),
    ]

    if train and resolved_augmentation["enabled"]:
        if resolved_augmentation["rotation_deg"] > 0:
            transform_steps.append(transforms.RandomRotation(resolved_augmentation["rotation_deg"]))
        if any(value > 0 for value in resolved_augmentation["translate"]):
            transform_steps.append(
                transforms.RandomAffine(
                    degrees=0,
                    translate=resolved_augmentation["translate"],
                )
            )
        if resolved_augmentation["gaussian_noise_std"] > 0:
            transform_steps.append(GaussianNoise(resolved_augmentation["gaussian_noise_std"]))

    transform_steps.append(transforms.Normalize((0.1307,), (0.3081,)))
    return transforms.Compose(transform_steps)


def _raw_files_ready() -> bool:
    return all((EMNIST_RAW_DIR / file_name).exists() for file_name in BALANCED_FILES)


def _find_source_file(source_dir: Path, file_name: str) -> Path | None:
    for candidate in source_dir.rglob(file_name):
        if candidate.is_file():
            return candidate
    return None


def _copy_or_extract(source_path: Path, target_path: Path) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    if source_path.suffix == ".gz":
        with gzip.open(source_path, "rb") as src, target_path.open("wb") as dst:
            shutil.copyfileobj(src, dst)
        return

    shutil.copy2(source_path, target_path)


def download_emnist(force_sync: bool = False) -> Path:
    if _raw_files_ready() and not force_sync:
        return EMNIST_RAW_DIR

    try:
        import kagglehub
    except ImportError as exc:
        raise ImportError("kagglehub is required. Run `pip install -r requirements.txt`.") from exc

    source_dir = Path(kagglehub.dataset_download(KAGGLE_DATASET))
    ensure_dir(EMNIST_RAW_DIR)

    for file_name in BALANCED_FILES:
        target_path = EMNIST_RAW_DIR / file_name
        if target_path.exists() and not force_sync:
            continue

        source_path = _find_source_file(source_dir, file_name)
        if source_path is None:
            source_path = _find_source_file(source_dir, f"{file_name}.gz")
        if source_path is None:
            raise FileNotFoundError(f"Cannot find `{file_name}` in downloaded dataset: {source_dir}")

        _copy_or_extract(source_path, target_path)

    return EMNIST_RAW_DIR


def _subset_training_dataset(dataset, train_subset_ratio: float, seed: int):
    if not 0 < train_subset_ratio <= 1.0:
        raise ValueError(f"train_subset_ratio must be in (0, 1], got {train_subset_ratio}.")
    if train_subset_ratio >= 1.0:
        return dataset

    subset_size = max(1, int(len(dataset) * train_subset_ratio))
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator)[:subset_size].tolist()
    return Subset(dataset, indices)


def build_dataloaders(
    batch_size: int = 128,
    val_ratio: float = 0.1,
    num_workers: int = 2,
    seed: int = 42,
    train_subset_ratio: float = 1.0,
    augmentation_config: dict | None = None,
):
    download_emnist()

    train_transform = build_transform(train=True, augmentation_config=augmentation_config)
    eval_transform = build_transform(train=False, augmentation_config=None)
    train_source_dataset = datasets.EMNIST(
        root=str(TORCHVISION_ROOT),
        split="balanced",
        train=True,
        download=False,
        transform=train_transform,
    )
    test_dataset = datasets.EMNIST(
        root=str(TORCHVISION_ROOT),
        split="balanced",
        train=False,
        download=False,
        transform=eval_transform,
    )

    val_source_dataset = datasets.EMNIST(
        root=str(TORCHVISION_ROOT),
        split="balanced",
        train=True,
        download=False,
        transform=eval_transform,
    )

    val_size = max(1, int(len(train_source_dataset) * val_ratio))
    train_size = len(train_source_dataset) - val_size
    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(
        range(len(train_source_dataset)),
        [train_size, val_size],
        generator=generator,
    )
    train_dataset = Subset(train_source_dataset, train_dataset.indices)
    val_dataset = Subset(val_source_dataset, val_dataset.indices)
    train_dataset = _subset_training_dataset(train_dataset, train_subset_ratio=train_subset_ratio, seed=seed)

    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader, test_loader, len(test_dataset.classes)
