import json
from pathlib import Path

import pandas as pd

from churn.config import Config
from churn.predict import load_model, predict
from churn.train import run_selection, run_training


def test_run_selection_ranks_candidates(config: Config) -> None:
    table = run_selection(config)

    assert set(table.index) == set(config.selection)
    assert table["mean_test_recall"].is_monotonic_decreasing
    assert Path(config.artifacts.selection_path).exists()


def test_run_training_saves_model_and_metrics(config: Config) -> None:
    metrics = run_training(config)

    assert 0.0 <= metrics["test"]["recall"] <= 1.0
    assert metrics["best_params"]
    saved = json.loads(Path(config.artifacts.metrics_path).read_text(encoding="utf-8"))
    assert saved == metrics
    assert Path(config.artifacts.model_path).exists()


def test_predict_on_raw_data_without_target(config: Config, churn_df: pd.DataFrame) -> None:
    run_training(config)
    model = load_model(config.artifacts.model_path)
    raw = churn_df.drop(columns="Exited").head(5)

    result = predict(model, raw, threshold=0.5)

    assert result.columns.tolist() == ["id", "churn_proba", "churn_pred"]
    assert result["churn_proba"].between(0, 1).all()
    assert set(result["churn_pred"]) <= {0, 1}
