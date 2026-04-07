from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from time import perf_counter

import torch
from torch import nn
from torch.optim import lr_scheduler

from utils import ensure_dir, save_json


def resolve_optimizer_config(train_config: dict) -> dict:
    optimizer_config = train_config.get("optimizer", {})
    if isinstance(optimizer_config, str):
        optimizer_config = {"name": optimizer_config}

    resolved = {
        "name": str(optimizer_config.get("name", train_config.get("optimizer", "adam"))).lower(),
        "lr": float(optimizer_config.get("lr", train_config.get("lr", 1e-3))),
        "weight_decay": float(optimizer_config.get("weight_decay", train_config.get("weight_decay", 0.0))),
        "momentum": float(optimizer_config.get("momentum", train_config.get("momentum", 0.9))),
        "alpha": float(optimizer_config.get("alpha", 0.99)),
        "eps": float(optimizer_config.get("eps", 1e-8)),
        "rho": float(optimizer_config.get("rho", 0.9)),
        "lambd": float(optimizer_config.get("lambd", 1e-4)),
        "t0": float(optimizer_config.get("t0", 1e6)),
        "nesterov": bool(optimizer_config.get("nesterov", False)),
    }

    regularization_config = resolve_regularization_config(train_config)
    if regularization_config["type"] == "l2" and regularization_config["l2_lambda"] > 0:
        resolved["weight_decay"] = regularization_config["l2_lambda"]

    return resolved


def resolve_scheduler_config(train_config: dict) -> dict:
    scheduler_config = train_config.get("scheduler", {})
    if isinstance(scheduler_config, str):
        scheduler_config = {"name": scheduler_config}

    return {
        "name": str(scheduler_config.get("name", "none")).lower(),
        "step_size": int(scheduler_config.get("step_size", 10)),
        "gamma": float(scheduler_config.get("gamma", 0.1)),
        "t_max": int(scheduler_config.get("t_max", train_config.get("epochs", 20))),
        "patience": int(scheduler_config.get("patience", 3)),
        "factor": float(scheduler_config.get("factor", 0.5)),
        "min_lr": float(scheduler_config.get("min_lr", 0.0)),
    }


def resolve_regularization_config(train_config: dict) -> dict:
    regularization_config = train_config.get("regularization", {})
    if isinstance(regularization_config, str):
        regularization_config = {"type": regularization_config}

    return {
        "type": str(regularization_config.get("type", "none")).lower(),
        "l1_lambda": float(regularization_config.get("l1_lambda", 0.0)),
        "l2_lambda": float(regularization_config.get("l2_lambda", 0.0)),
    }


def build_optimizer(model, optimizer_config: dict):
    name = optimizer_config["name"]
    lr = optimizer_config["lr"]
    weight_decay = optimizer_config["weight_decay"]

    if name == "adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            eps=optimizer_config["eps"],
        )
    if name == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            eps=optimizer_config["eps"],
        )
    if name == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=optimizer_config["momentum"],
            weight_decay=weight_decay,
            nesterov=optimizer_config["nesterov"],
        )
    if name == "rmsprop":
        return torch.optim.RMSprop(
            model.parameters(),
            lr=lr,
            momentum=optimizer_config["momentum"],
            weight_decay=weight_decay,
            alpha=optimizer_config["alpha"],
            eps=optimizer_config["eps"],
        )
    if name == "asgd":
        return torch.optim.ASGD(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            lambd=optimizer_config["lambd"],
            t0=optimizer_config["t0"],
        )
    if name == "adagrad":
        return torch.optim.Adagrad(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            eps=optimizer_config["eps"],
        )
    raise ValueError(f"Unsupported optimizer: {name}")


def build_scheduler(optimizer, scheduler_config: dict):
    name = scheduler_config["name"]
    if name == "none":
        return None
    if name == "step":
        return lr_scheduler.StepLR(
            optimizer,
            step_size=scheduler_config["step_size"],
            gamma=scheduler_config["gamma"],
        )
    if name == "cosine":
        return lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=scheduler_config["t_max"],
            eta_min=scheduler_config["min_lr"],
        )
    if name in {"plateau", "reduce_on_plateau", "reduce_lr_on_plateau"}:
        return lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=scheduler_config["factor"],
            patience=scheduler_config["patience"],
            min_lr=scheduler_config["min_lr"],
        )
    if name == "exponential":
        return lr_scheduler.ExponentialLR(
            optimizer,
            gamma=scheduler_config["gamma"],
        )
    raise ValueError(f"Unsupported scheduler: {name}")


def compute_l1_penalty(model) -> torch.Tensor:
    penalty = None
    for parameter in model.parameters():
        if not parameter.requires_grad:
            continue
        value = parameter.abs().sum()
        penalty = value if penalty is None else penalty + value

    if penalty is None:
        return torch.tensor(0.0)
    return penalty


def run_epoch(model, data_loader, criterion, optimizer=None, device: str = "cpu", l1_lambda: float = 0.0) -> dict:
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
            if training and l1_lambda > 0:
                loss = loss + l1_lambda * compute_l1_penalty(model)
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
    device: str,
    save_dir: str | Path,
    train_config: dict,
) -> dict:
    save_dir = ensure_dir(save_dir)
    epochs = int(train_config.get("epochs", 20))
    optimizer_config = resolve_optimizer_config(train_config)
    scheduler_config = resolve_scheduler_config(train_config)
    regularization_config = resolve_regularization_config(train_config)

    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(model=model, optimizer_config=optimizer_config)
    scheduler = build_scheduler(optimizer, scheduler_config)

    history: list[dict] = []
    best_val_accuracy = -1.0
    best_epoch = 0
    best_state = deepcopy(model.state_dict())
    start_time = perf_counter()

    for epoch in range(1, epochs + 1):
        current_lr = optimizer.param_groups[0]["lr"]
        train_metrics = run_epoch(
            model,
            train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            l1_lambda=regularization_config["l1_lambda"] if regularization_config["type"] == "l1" else 0.0,
        )
        val_metrics = run_epoch(model, val_loader, criterion=criterion, optimizer=None, device=device)

        if scheduler is not None:
            if scheduler_config["name"] in {"plateau", "reduce_on_plateau", "reduce_lr_on_plateau"}:
                scheduler.step(val_metrics["loss"])
            else:
                scheduler.step()

        record = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "lr": current_lr,
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
        "optimizer": optimizer_config,
        "scheduler": scheduler_config,
        "regularization": regularization_config,
        "device": device,
        "training_seconds": duration,
    }
    save_json(summary, save_dir / "summary.json")
    return summary
