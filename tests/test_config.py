from pathlib import Path

import pytest

from churn.config import load_config

from .conftest import PROJECT_ROOT


def test_load_project_config() -> None:
    cfg = load_config(PROJECT_ROOT / "configs" / "config.yaml")

    assert cfg.data.target == "Exited"
    assert cfg.features.drop == ["id", "CustomerId", "Surname"]
    assert cfg.cv.refit_metric == "recall"
    assert cfg.tuning.model in cfg.selection


def test_missing_section_raises(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("data: {train_path: x.csv, target: y}\n", encoding="utf-8")

    with pytest.raises(KeyError):
        load_config(path)
