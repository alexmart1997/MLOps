"""Отбор моделей и обучение финальной модели."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from optuna.distributions import (
    BaseDistribution,
    CategoricalDistribution,
    FloatDistribution,
    IntDistribution,
)
from optuna_integration import OptunaSearchCV
from sklearn.base import BaseEstimator
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from churn.config import Config
from churn.data import load_dataset, split_train_test
from churn.pipeline import build_pipeline

logger = logging.getLogger(__name__)

SCORING = ("recall", "precision", "f1")


def _make_cv(cfg: Config) -> StratifiedKFold:
    return StratifiedKFold(
        n_splits=cfg.cv.n_splits,
        shuffle=cfg.cv.shuffle,
        random_state=cfg.cv.random_state if cfg.cv.shuffle else None,
    )


def _load_split(cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    required = [cfg.data.target, *cfg.features.categorical, *cfg.features.numeric]
    df = load_dataset(cfg.data.train_path, required_columns=required)
    return split_train_test(df, cfg.data.target, cfg.data.test_size, cfg.data.random_state)


def _to_distribution(spec: dict[str, Any]) -> BaseDistribution:
    kind = spec["type"]
    if kind == "int":
        return IntDistribution(spec["low"], spec["high"], step=spec.get("step", 1))
    if kind == "float":
        return FloatDistribution(spec["low"], spec["high"], log=spec.get("log", False))
    if kind == "categorical":
        return CategoricalDistribution(spec["choices"])
    raise ValueError(f"Unknown distribution type {kind!r}")


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def evaluate(model: BaseEstimator, x: pd.DataFrame, y: pd.Series) -> dict[str, float]:
    """Метрики качества на отложенной выборке."""
    y_pred = model.predict(x)
    y_proba = model.predict_proba(x)[:, 1]
    return {
        "recall": float(recall_score(y, y_pred)),
        "precision": float(precision_score(y, y_pred, zero_division=0)),
        "f1": float(f1_score(y, y_pred)),
        "roc_auc": float(roc_auc_score(y, y_proba)),
    }


def run_selection(cfg: Config) -> pd.DataFrame:
    """GridSearchCV по всем кандидатам из `cfg.selection`.

    Возвращает таблицу «модель × метрики CV», отсортированную по ключевой метрике.
    """
    x_train, _, y_train, _ = _load_split(cfg)
    cv = _make_cv(cfg)
    rows: dict[str, dict[str, Any]] = {}

    for name, grid in cfg.selection.items():
        logger.info("Grid search for %s (%d params)", name, len(grid))
        search = GridSearchCV(
            build_pipeline(name, cfg.features, cfg.data.random_state),
            grid,
            cv=cv,
            scoring=list(SCORING),
            refit=cfg.cv.refit_metric,
            return_train_score=True,
            n_jobs=cfg.n_jobs,
        )
        search.fit(x_train, y_train)
        results = search.cv_results_
        row: dict[str, Any] = {
            f"mean_{split}_{metric}": float(results[f"mean_{split}_{metric}"][search.best_index_])
            for split in ("test", "train")
            for metric in SCORING
        }
        row["refit_time_sec"] = float(search.refit_time_)
        row["best_params"] = json.dumps(search.best_params_, default=str)
        rows[name] = row

    table = pd.DataFrame.from_dict(rows, orient="index").sort_values(
        f"mean_test_{cfg.cv.refit_metric}", ascending=False
    )
    path = Path(cfg.artifacts.selection_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index_label="model")
    logger.info("Model selection results saved to %s\n%s", path, table.drop(columns="best_params"))
    return table


def run_training(cfg: Config) -> dict[str, Any]:
    """Подбор гиперпараметров финальной модели Optuna, оценка на тесте, сохранение."""
    x_train, x_test, y_train, y_test = _load_split(cfg)
    tuning = cfg.tuning

    search = OptunaSearchCV(
        build_pipeline(
            tuning.model, cfg.features, cfg.data.random_state, drop_low_importance=True
        ),
        {name: _to_distribution(spec) for name, spec in tuning.search_space.items()},
        cv=_make_cv(cfg),
        scoring=cfg.cv.refit_metric,
        n_trials=tuning.n_trials,
        return_train_score=True,
        n_jobs=cfg.n_jobs,
        random_state=tuning.random_state,
        verbose=0,
    )
    logger.info("Tuning %s with Optuna: %d trials", tuning.model, tuning.n_trials)
    search.fit(x_train, y_train)

    model = search.best_estimator_
    cv_results = search.cv_results_
    metrics: dict[str, Any] = {
        "model": tuning.model,
        "best_params": search.best_params_,
        "cv": {
            "mean_test_score": float(cv_results["mean_test_score"][search.best_index_]),
            "mean_train_score": float(cv_results["mean_train_score"][search.best_index_]),
        },
        "test": evaluate(model, x_test, y_test),
        "n_train": len(x_train),
        "n_test": len(x_test),
    }
    # Нормализуем типы (numpy → python), чтобы метрики совпадали с сохранённым JSON.
    metrics = json.loads(json.dumps(metrics, default=str))

    model_path = Path(cfg.artifacts.model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    _write_json(cfg.artifacts.metrics_path, metrics)
    logger.info("Model saved to %s; test metrics: %s", model_path, metrics["test"])
    return metrics
