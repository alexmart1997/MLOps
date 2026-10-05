"""Загрузка и типизация конфигурации проекта из YAML."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

ParamGrid = dict[str, list[Any]]


@dataclass
class DataConfig:
    train_path: str
    target: str
    test_size: float = 0.2
    random_state: int = 42


@dataclass
class FeaturesConfig:
    drop: list[str]
    categorical: list[str]
    numeric: list[str]
    low_importance: list[str] = field(default_factory=list)


@dataclass
class CVConfig:
    n_splits: int = 3
    shuffle: bool = True
    random_state: int = 52
    refit_metric: str = "recall"


@dataclass
class TuningConfig:
    model: str
    search_space: dict[str, dict[str, Any]]
    n_trials: int = 10
    random_state: int = 42


@dataclass
class ArtifactsConfig:
    model_path: str
    metrics_path: str
    selection_path: str


@dataclass
class Config:
    data: DataConfig
    features: FeaturesConfig
    cv: CVConfig
    selection: dict[str, ParamGrid]
    tuning: TuningConfig
    artifacts: ArtifactsConfig
    n_jobs: int = -1


def load_config(path: str | Path) -> Config:
    """Читает YAML-конфиг; отсутствие обязательной секции — KeyError."""
    with Path(path).open(encoding="utf-8") as fh:
        raw: dict[str, Any] = yaml.safe_load(fh)

    return Config(
        data=DataConfig(**raw["data"]),
        features=FeaturesConfig(**raw["features"]),
        cv=CVConfig(**raw["cv"]),
        selection=raw["selection"],
        tuning=TuningConfig(**raw["tuning"]),
        artifacts=ArtifactsConfig(**raw["artifacts"]),
        n_jobs=raw.get("n_jobs", -1),
    )
