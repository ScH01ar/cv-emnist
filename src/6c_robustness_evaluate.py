from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torchvision.transforms.functional as TF

from dataset import (
    EMNISTOrientationCorrection,
    GaussianNoise,
    TORCHVISION_ROOT,
    download_emnist,
)
from main import build_model, load_config
from utils import PROJECT_ROOT, ensure_dir, get_device, set_seed


MODEL_NAMES = ["mlp", "cnn", "resnet", "vit"]

PERTURBATIONS = [
    "clean",
    "rotation_5",
    "rotation_10",
    "rotation_15",
    "gaussian_noise_0.05",
    "gaussian_noise_0.10",
    "gaussian_noise_0.15",
    "gaussian_noise_0.20",
    "blur_3",
    "blur_5",
]


def build_eval_transform(kind: str):
    steps = [
        transforms.ToTensor(),
        EMNISTOrientationCorrection(),
    ]

    if kind == "clean":
        pass

    elif kind.startswith("rotation_"):
        angle = float(kind.split("_")[1])
        steps.append(
            transforms.Lambda(
                lambda x, a=angle: TF.rotate(x, angle=a, fill=0.0)
            )
        )

    elif kind.startswith("gaussian_noise_"):
        std = float(kind.split("_")[-1])
        steps.append(GaussianNoise(std=std))

    elif kind.startswith("blur_"):
        kernel_size = int(kind.split("_")[1])
        steps.append(transforms.GaussianBlur(kernel_size=kernel_size, sigma=1.0))

    else:
        raise ValueError(f"Unknown robustness transform: {kind}")

    steps.append(transforms.Normalize((0.1307,), (0.3081,)))
    return transforms.Compose(steps)


def build_test_loader(kind: str, batch_size: int, num_workers: int):
    download_emnist()

    dataset = datasets.EMNIST(
        root=str(TORCHVISION_ROOT),
        split="balanced",
        train=False,
        download=False,
        transform=build_eval_transform(kind),
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return loader, len(dataset.classes)


def unwrap_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            return checkpoint["model_state_dict"]
        if "state_dict" in checkpoint:
            return checkpoint["state_dict"]
    return checkpoint


def load_best_model(model_name: str, num_classes: int, device: str):
    run_dir = PROJECT_ROOT / "artifacts" / "best_runs" / model_name
    config_path = run_dir / "config.yaml"
    best_path = run_dir / "best.pt"

    if not config_path.exists():
        raise FileNotFoundError(f"Missing config: {config_path}")
    if not best_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {best_path}")

    config = load_config(config_path)
    model_config = config["model"]

    model = build_model(
        model_file=model_config["file"],
        class_name=model_config["class_name"],
        model_kwargs=model_config.get("kwargs", {}),
        num_classes=num_classes,
    ).to(device)

    checkpoint = torch.load(best_path, map_location=device)
    state_dict = unwrap_state_dict(checkpoint)
    model.load_state_dict(state_dict)
    model.eval()

    return model


def evaluate_full_metrics(model, data_loader, device: str, num_classes: int):
    total_loss = 0.0
    total_examples = 0
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.long)

    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            loss = F.cross_entropy(outputs, targets, reduction="sum")
            preds = outputs.argmax(dim=1)

            total_loss += loss.item()
            total_examples += targets.size(0)

            for target, pred in zip(targets.cpu(), preds.cpu()):
                confusion[target.long(), pred.long()] += 1

    tp = confusion.diag().float()
    support = confusion.sum(dim=1).float()
    predicted = confusion.sum(dim=0).float()

    precision_per_class = tp / predicted.clamp(min=1)
    recall_per_class = tp / support.clamp(min=1)
    f1_per_class = 2 * precision_per_class * recall_per_class / (
        precision_per_class + recall_per_class
    ).clamp(min=1e-12)

    accuracy = tp.sum().item() / max(1, confusion.sum().item())

    return {
        "loss": total_loss / max(1, total_examples),
        "accuracy": accuracy,
        "precision_macro": precision_per_class.mean().item(),
        "recall_macro": recall_per_class.mean().item(),
        "f1_macro": f1_per_class.mean().item(),
    }


def save_results_csv(rows: list[dict], output_path: Path):
    fieldnames = [
        "model",
        "perturbation",
        "loss",
        "accuracy",
        "precision_macro",
        "recall_macro",
        "f1_macro",
        "accuracy_drop",
        "relative_accuracy_drop",
    ]

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_average_drop_csv(rows: list[dict], output_path: Path):
    fieldnames = [
        "model",
        "avg_drop_all",
        "avg_drop_rotation",
        "avg_drop_noise",
        "avg_drop_blur",
        "max_drop",
    ]

    output_rows = []

    for model_name in MODEL_NAMES:
        model_rows = [
            row for row in rows
            if row["model"] == model_name and row["perturbation"] != "clean"
        ]
        rotation_rows = [row for row in model_rows if row["perturbation"].startswith("rotation")]
        noise_rows = [row for row in model_rows if row["perturbation"].startswith("gaussian_noise")]
        blur_rows = [row for row in model_rows if row["perturbation"].startswith("blur")]

        def avg_drop(selected):
            if not selected:
                return 0.0
            return sum(row["accuracy_drop"] for row in selected) / len(selected)

        output_rows.append(
            {
                "model": model_name,
                "avg_drop_all": round(avg_drop(model_rows), 6),
                "avg_drop_rotation": round(avg_drop(rotation_rows), 6),
                "avg_drop_noise": round(avg_drop(noise_rows), 6),
                "avg_drop_blur": round(avg_drop(blur_rows), 6),
                "max_drop": round(max(row["accuracy_drop"] for row in model_rows), 6),
            }
        )

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)


def save_plots(rows: list[dict], output_dir: Path):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skip plots.")
        return

    perturbations = [p for p in PERTURBATIONS if p != "clean"]
    models = MODEL_NAMES

    def get_value(model_name: str, perturbation: str, field: str) -> float:
        for row in rows:
            if row["model"] == model_name and row["perturbation"] == perturbation:
                return row[field]
        return 0.0

    x = list(range(len(perturbations)))
    width = 0.18

    # Plot 1: accuracy under perturbations
    fig, ax = plt.subplots(figsize=(13, 5))
    for i, model_name in enumerate(models):
        values = [get_value(model_name, p, "accuracy") for p in perturbations]
        offsets = [value + (i - 1.5) * width for value in x]
        ax.bar(offsets, values, width=width, label=model_name)

    ax.set_xticks(x)
    ax.set_xticklabels(perturbations, rotation=30, ha="right")
    ax.set_ylabel("Accuracy")
    ax.set_title("Robustness: accuracy under perturbations")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "robustness_accuracy.png", dpi=200)
    plt.close(fig)

    # Plot 2: accuracy drop under perturbations
    fig, ax = plt.subplots(figsize=(13, 5))
    for i, model_name in enumerate(models):
        values = [get_value(model_name, p, "accuracy_drop") for p in perturbations]
        offsets = [value + (i - 1.5) * width for value in x]
        ax.bar(offsets, values, width=width, label=model_name)

    ax.set_xticks(x)
    ax.set_xticklabels(perturbations, rotation=30, ha="right")
    ax.set_ylabel("Accuracy drop")
    ax.set_title("Robustness: accuracy drop under perturbations")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "robustness_accuracy_drop.png", dpi=200)
    plt.close(fig)

    # Plot 3: average drop by model
    avg_rows = []
    for model_name in models:
        model_rows = [
            row for row in rows
            if row["model"] == model_name and row["perturbation"] != "clean"
        ]
        avg_rows.append(
            sum(row["accuracy_drop"] for row in model_rows) / len(model_rows)
        )

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(models, avg_rows)
    ax.set_ylabel("Average accuracy drop")
    ax.set_title("Average robustness drop across perturbations")
    fig.tight_layout()
    fig.savefig(output_dir / "robustness_average_drop.png", dpi=200)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate robustness of best EMNIST models.")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output-dir", type=str, default="artifacts/6_c")
    return parser.parse_args()


def main():
    args = parse_args()

    set_seed(42)
    device = get_device(args.device)
    output_dir = ensure_dir(PROJECT_ROOT / args.output_dir)

    loaders = {}
    num_classes = None

    print("Preparing perturbed test loaders...")
    for perturbation in PERTURBATIONS:
        loader, current_num_classes = build_test_loader(
            kind=perturbation,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        loaders[perturbation] = loader

        if num_classes is None:
            num_classes = current_num_classes
        elif num_classes != current_num_classes:
            raise ValueError("Inconsistent class count across test loaders.")

    print(f"Device: {device}")
    print(f"Num classes: {num_classes}")
    print(f"Output dir: {output_dir}")

    rows = []

    for model_name in MODEL_NAMES:
        print()
        print(f"===== Loading {model_name} =====")
        model = load_best_model(model_name=model_name, num_classes=num_classes, device=device)

        clean_accuracy = None

        for perturbation in PERTURBATIONS:
            print(f"Evaluating {model_name} on {perturbation}...")

            torch.manual_seed(42)

            metrics = evaluate_full_metrics(
                model=model,
                data_loader=loaders[perturbation],
                device=device,
                num_classes=num_classes,
            )

            if perturbation == "clean":
                clean_accuracy = metrics["accuracy"]

            accuracy_drop = clean_accuracy - metrics["accuracy"]
            relative_drop = accuracy_drop / max(clean_accuracy, 1e-12)

            row = {
                "model": model_name,
                "perturbation": perturbation,
                "loss": round(metrics["loss"], 6),
                "accuracy": round(metrics["accuracy"], 6),
                "precision_macro": round(metrics["precision_macro"], 6),
                "recall_macro": round(metrics["recall_macro"], 6),
                "f1_macro": round(metrics["f1_macro"], 6),
                "accuracy_drop": round(accuracy_drop, 6),
                "relative_accuracy_drop": round(relative_drop, 6),
            }
            rows.append(row)

            print(
                f"{model_name:6s} | {perturbation:20s} | "
                f"acc={row['accuracy']:.4f} | "
                f"drop={row['accuracy_drop']:.4f} | "
                f"f1={row['f1_macro']:.4f}"
            )

    csv_path = output_dir / "robustness_results.csv"
    avg_csv_path = output_dir / "robustness_average_drop.csv"
    json_path = output_dir / "robustness_summary.json"

    save_results_csv(rows, csv_path)
    save_average_drop_csv(rows, avg_csv_path)

    with json_path.open("w", encoding="utf-8") as file:
        json.dump({"perturbations": PERTURBATIONS, "results": rows}, file, indent=2)

    save_plots(rows, output_dir)

    print()
    print(f"Saved CSV:      {csv_path}")
    print(f"Saved avg CSV:  {avg_csv_path}")
    print(f"Saved JSON:     {json_path}")
    print(f"Saved plots to: {output_dir}")


if __name__ == "__main__":
    main()
