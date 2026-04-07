from __future__ import annotations

import gzip
import shutil
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split
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


def build_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Lambda(lambda x: torch.rot90(x, -1, dims=[1, 2])),
            transforms.Lambda(lambda x: torch.flip(x, dims=[1])),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )


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


def build_dataloaders(
    batch_size: int = 128,
    val_ratio: float = 0.1,
    num_workers: int = 2,
    seed: int = 42,
):
    download_emnist()

    transform = build_transform()
    train_dataset = datasets.EMNIST(
        root=str(TORCHVISION_ROOT),
        split="balanced",
        train=True,
        download=False,
        transform=transform,
    )
    test_dataset = datasets.EMNIST(
        root=str(TORCHVISION_ROOT),
        split="balanced",
        train=False,
        download=False,
        transform=transform,
    )

    val_size = max(1, int(len(train_dataset) * val_ratio))
    train_size = len(train_dataset) - val_size
    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size], generator=generator)

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
