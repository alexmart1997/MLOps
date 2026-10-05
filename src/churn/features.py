"""Преобразования признаков, выбранные по итогам EDA (см. notebooks/Case_1.ipynb)."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd


def drop_columns(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    """Удаляет колонки; отсутствующие пропускает, чтобы на инференсе
    можно было подавать данные без технических полей."""
    return df.drop(columns=list(columns), errors="ignore")


def add_has_balance(df: pd.DataFrame) -> pd.DataFrame:
    """Balance — zero-inflated признак: добавляем бинарный флаг ненулевого баланса."""
    result = df.copy()
    result["HasBalance"] = (result["Balance"] > 0).astype(int)
    return result
