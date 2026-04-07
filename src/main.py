from __future__ import annotations

import argparse
import inspect

from dataset import build_dataloaders
from trainer import evaluate_model, train_model
from utils import PROJECT_ROOT, get_device, load_class_from_file, load_yaml, save_json, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Minimal EMNIST training entrypoint.")
    parser.add_argument("--config", required=True, help="Path to a YAML config file.")
    return parser.parse_args()


def build_model(model_file: str, class_name: str, model_kwargs: dict, num_classes: int):
    model_path = PROJECT_ROOT / model_file
    model_class = load_class_from_file(model_path, class_name)
    kwargs = dict(model_kwargs)

    signature = inspect.signature(model_class)
    if "num_classes" in signature.parameters and "num_classes" not in kwargs:
        kwargs["num_classes"] = num_classes

    return model_class(**kwargs)


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    seed = int(config.get("seed", 42))
    set_seed(seed)

    data_config = config.get("data", {})
    model_config = config.get("model", {})
    train_config = config.get("train", {})

    train_loader, val_loader, test_loader, num_classes = build_dataloaders(
        batch_size=int(data_config.get("batch_size", 128)),
        val_ratio=float(data_config.get("val_ratio", 0.1)),
        num_workers=int(data_config.get("num_workers", 2)),
        seed=seed,
    )
    device = get_device(str(config.get("device", "auto")))
    model = build_model(
        model_file=model_config["file"],
        class_name=model_config["class_name"],
        model_kwargs=model_config.get("kwargs", {}),
        num_classes=num_classes,
    ).to(device)

    run_name = train_config["run_name"]
    run_dir = PROJECT_ROOT / "runs" / run_name
    print(f"Device: {device}")
    print(f"Run dir: {run_dir}")

    summary = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=int(train_config.get("epochs", 20)),
        lr=float(train_config.get("lr", 1e-3)),
        device=device,
        save_dir=run_dir,
        optimizer_name=str(train_config.get("optimizer", "adam")),
        weight_decay=float(train_config.get("weight_decay", 0.0)),
        momentum=float(train_config.get("momentum", 0.9)),
    )
    test_metrics = evaluate_model(model, test_loader, device=device)
    save_json(test_metrics, run_dir / "test_metrics.json")

    print(
        "Finished | "
        f"best_epoch={summary['best_epoch']} | "
        f"best_val_acc={summary['best_val_accuracy']:.4f} | "
        f"test_acc={test_metrics['accuracy']:.4f}"
    )


if __name__ == "__main__":
    main()
