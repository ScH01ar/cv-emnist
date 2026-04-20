from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path

DEFAULT_RATIOS = (0.3, 0.5, 1.0)


def parse_args():
    parser = argparse.ArgumentParser(description="Run small-data experiments from a base YAML config.")
    parser.add_argument("--config", required=True, help="Path to the base YAML config.")
    parser.add_argument(
        "--ratios",
        nargs="+",
        type=float,
        default=list(DEFAULT_RATIOS),
        help="Training subset ratios to run, e.g. --ratios 0.3 0.5 1.0",
    )
    return parser.parse_args()


def _ratio_suffix(ratio: float) -> str:
    return str(int(round(ratio * 100)))


def build_small_data_config(base_config: dict, ratio: float) -> dict:
    if not 0 < ratio <= 1.0:
        raise ValueError(f"Each ratio must be in (0, 1], got {ratio}.")

    config = deepcopy(base_config)
    data_config = config.setdefault("data", {})
    train_config = config.setdefault("train", {})
    base_run_name = str(train_config["run_name"])

    data_config["train_subset_ratio"] = ratio
    train_config["run_name"] = f"{base_run_name}_small_{_ratio_suffix(ratio)}"
    return config


def main() -> None:
    args = parse_args()
    from main import load_config, run_config

    base_config = load_config(args.config)
    base_run_name = str(base_config.get("train", {}).get("run_name", "experiment"))
    records = []

    for ratio in args.ratios:
        config = build_small_data_config(base_config, ratio)
        result = run_config(config)
        records.append(
            {
                "ratio": ratio,
                "run_name": result["run_name"],
                "best_epoch": result["summary"]["best_epoch"],
                "best_val_accuracy": result["summary"]["best_val_accuracy"],
                "test_accuracy": result["test_metrics"]["accuracy"],
                "test_loss": result["test_metrics"]["loss"],
            }
        )

    project_root = Path(__file__).resolve().parents[1]
    output_path = project_root / "runs" / f"{base_run_name}_small_data_summary.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump({"base_run_name": base_run_name, "results": records}, handle, indent=2, ensure_ascii=False)
    print(f"Saved small-data summary to: {output_path}")


if __name__ == "__main__":
    main()
