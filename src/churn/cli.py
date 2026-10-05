"""Командная строка: `churn select | train | predict`."""

from __future__ import annotations

import argparse
import logging
import os
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from churn.config import load_config
from churn.predict import load_model, predict
from churn.train import run_selection, run_training

DEFAULT_CONFIG = Path("configs/config.yaml")


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="churn", description="Bank churn prediction")
    parser.add_argument("-c", "--config", type=Path, default=DEFAULT_CONFIG)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("select", help="compare candidate models with GridSearchCV")
    sub.add_parser("train", help="tune the final model with Optuna and save it")

    pred = sub.add_parser("predict", help="score a CSV with the saved model")
    pred.add_argument("input", type=Path, help="CSV with raw features")
    pred.add_argument("-o", "--output", type=Path, default=Path("reports/predictions.csv"))
    pred.add_argument("--threshold", type=float, default=0.5)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = _parse_args(argv)
    cfg = load_config(args.config)

    if args.command == "select":
        run_selection(cfg)
    elif args.command == "train":
        run_training(cfg)
    elif args.command == "predict":
        model = load_model(cfg.artifacts.model_path)
        result = predict(model, pd.read_csv(args.input), threshold=args.threshold)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(args.output, index=False)
        logging.getLogger(__name__).info("Predictions saved to %s", args.output)


if __name__ == "__main__":
    main()
