"""Инференс сохранённой модели."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline


def load_model(path: str | Path) -> Pipeline:
    model = joblib.load(path)
    if not isinstance(model, Pipeline):
        raise TypeError(f"Expected sklearn Pipeline in {path}, got {type(model).__name__}")
    return model


def predict(
    model: Pipeline, df: pd.DataFrame, threshold: float = 0.5, id_column: str = "id"
) -> pd.DataFrame:
    """Вероятность оттока и бинарный прогноз для каждой строки сырых данных."""
    proba = model.predict_proba(df)[:, 1]
    ids = df[id_column] if id_column in df.columns else pd.Series(df.index, index=df.index)
    return pd.DataFrame(
        {
            "id": ids.to_numpy(),
            "churn_proba": proba,
            "churn_pred": (proba >= threshold).astype(int),
        }
    )
