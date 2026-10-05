"""Загрузка датасета и разбиение на train/test."""

from __future__ import annotations

import logging
from collections.abc import Iterable
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


def load_dataset(path: str | Path, required_columns: Iterable[str] = ()) -> pd.DataFrame:
    """Читает CSV (путь или URL) и проверяет наличие обязательных колонок."""
    df = pd.read_csv(path)
    missing = sorted(set(required_columns) - set(df.columns))
    if missing:
        raise ValueError(f"Dataset {path} is missing required columns: {missing}")
    logger.info("Loaded dataset %s: %d rows, %d columns", path, *df.shape)
    return df


def split_train_test(
    df: pd.DataFrame, target: str, test_size: float, random_state: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Стратифицированный split по таргету: X_train, X_test, y_train, y_test."""
    x = df.drop(columns=[target])
    y = df[target]
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, stratify=y, random_state=random_state
    )
    return x_train, x_test, y_train, y_test
