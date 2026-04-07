from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from time import perf_counter

import torch
from torch import nn

from utils import ensure_dir, save_json


def build_optimizer(model, optimizer_name: str, lr: float, weight_decay: float = 0.0, momentum: float = 0.9):
    name = optimizer_name.lower()
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
        )
    raise ValueError(f"Unsupported optimizer: {optimizer_name}")


def run_epoch(model, data_loader, criterion, optimizer=None, device: str = "cpu") -> dict:
    training = optimizer is not None
    model.train(mode=training)

    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for inputs, targets in data_loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            if training:
                loss.backward()
                optimizer.step()

        batch_size = targets.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (outputs.argmax(dim=1) == targets).sum().item()
        total_examples += batch_size

    return {
        "loss": total_loss / total_examples,
        "accuracy": total_correct / total_examples,
    }


def evaluate_model(model, data_loader, device: str = "cpu") -> dict:
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        return run_epoch(model, data_loader, criterion=criterion, optimizer=None, device=device)


def train_model(
    model,
    train_loader,
    val_loader,
    epochs: int,
    lr: float,
    device: str,
    save_dir: str | Path,
    optimizer_name: str = "adam",
    weight_decay: float = 0.0,
    momentum: float = 0.9,
) -> dict:
    save_dir = ensure_dir(save_dir)
    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(
        model=model,
        optimizer_name=optimizer_name,
        lr=lr,
        weight_decay=weight_decay,
        momentum=momentum,
    )

    history: list[dict] = []
    best_val_accuracy = -1.0
    best_epoch = 0
    best_state = deepcopy(model.state_dict())
    start_time = perf_counter()

    for epoch in range(1, epochs + 1):
        train_metrics = run_epoch(model, train_loader, criterion=criterion, optimizer=optimizer, device=device)
        val_metrics = run_epoch(model, val_loader, criterion=criterion, optimizer=None, device=device)

        record = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
        }
        history.append(record)

        if val_metrics["accuracy"] >= best_val_accuracy:
            best_val_accuracy = val_metrics["accuracy"]
            best_epoch = epoch
            best_state = deepcopy(model.state_dict())
            torch.save(best_state, save_dir / "best.pt")

        print(
            f"Epoch {epoch:02d}/{epochs:02d} | "
            f"train_loss={train_metrics['loss']:.4f} | "
            f"train_acc={train_metrics['accuracy']:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | "
            f"val_acc={val_metrics['accuracy']:.4f}"
        )

    model.load_state_dict(best_state)
    duration = perf_counter() - start_time

    save_json({"history": history}, save_dir / "history.json")
    summary = {
        "epochs": epochs,
        "best_epoch": best_epoch,
        "best_val_accuracy": best_val_accuracy,
        "optimizer": optimizer_name,
        "learning_rate": lr,
        "weight_decay": weight_decay,
        "momentum": momentum,
        "device": device,
        "training_seconds": duration,
    }
    save_json(summary, save_dir / "summary.json")
    return summary
