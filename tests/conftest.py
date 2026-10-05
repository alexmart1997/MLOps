from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from churn.config import Config, load_config

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def make_churn_frame(n_rows: int = 400, seed: int = 0) -> pd.DataFrame:
    """Синтетический датасет со схемой Kaggle Playground S4E1."""
    rng = np.random.default_rng(seed)
    age = rng.integers(18, 92, n_rows).astype(float)
    num_products = rng.integers(1, 5, n_rows)
    is_active = rng.integers(0, 2, n_rows).astype(float)
    logit = 0.08 * (age - 40) - 0.9 * is_active + 0.6 * (num_products == 1) - 1.0
    exited = (rng.random(n_rows) < 1 / (1 + np.exp(-logit))).astype(int)
    balance = np.where(rng.random(n_rows) < 0.5, 0.0, rng.uniform(1e3, 2.5e5, n_rows))
    return pd.DataFrame(
        {
            "id": np.arange(n_rows),
            "CustomerId": rng.integers(15_500_000, 15_800_000, n_rows),
            "Surname": rng.choice(["A", "B", "C"], n_rows),
            "CreditScore": rng.integers(350, 851, n_rows),
            "Geography": rng.choice(["France", "Spain", "Germany"], n_rows),
            "Gender": rng.choice(["Male", "Female"], n_rows),
            "Age": age,
            "Tenure": rng.integers(0, 11, n_rows),
            "Balance": balance,
            "NumOfProducts": num_products,
            "HasCrCard": rng.integers(0, 2, n_rows).astype(float),
            "IsActiveMember": is_active,
            "EstimatedSalary": rng.uniform(11, 2e5, n_rows),
            "Exited": exited,
        }
    )


@pytest.fixture
def churn_df() -> pd.DataFrame:
    return make_churn_frame()


@pytest.fixture
def config(tmp_path: Path, churn_df: pd.DataFrame) -> Config:
    """Боевой конфиг, ужатый до быстрых сеток и путей во временной папке."""
    cfg = load_config(PROJECT_ROOT / "configs" / "config.yaml")
    data_path = tmp_path / "train.csv"
    churn_df.to_csv(data_path, index=False)
    cfg.data.train_path = str(data_path)
    cfg.selection = {
        "logreg": {"classifier__C": [0.1]},
        "catboost": {"classifier__iterations": [20], "classifier__depth": [3]},
    }
    cfg.tuning.n_trials = 2
    cfg.tuning.search_space = {
        "classifier__iterations": {"type": "int", "low": 10, "high": 30, "step": 10},
        "classifier__depth": {"type": "int", "low": 2, "high": 4, "step": 2},
        "classifier__auto_class_weights": {"type": "categorical", "choices": ["Balanced"]},
    }
    cfg.artifacts.model_path = str(tmp_path / "models" / "model.joblib")
    cfg.artifacts.metrics_path = str(tmp_path / "reports" / "metrics.json")
    cfg.artifacts.selection_path = str(tmp_path / "reports" / "selection.csv")
    cfg.n_jobs = 1
    return cfg
