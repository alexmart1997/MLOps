from pathlib import Path

import pandas as pd
import pytest

from churn.data import load_dataset, split_train_test


def test_load_dataset_reads_csv(tmp_path: Path, churn_df: pd.DataFrame) -> None:
    path = tmp_path / "train.csv"
    churn_df.to_csv(path, index=False)

    df = load_dataset(path, required_columns=["Age", "Exited"])

    assert df.shape == churn_df.shape


def test_load_dataset_fails_on_missing_columns(tmp_path: Path, churn_df: pd.DataFrame) -> None:
    path = tmp_path / "train.csv"
    churn_df.drop(columns=["Age"]).to_csv(path, index=False)

    with pytest.raises(ValueError, match="Age"):
        load_dataset(path, required_columns=["Age", "Exited"])


def test_split_is_stratified_and_drops_target(churn_df: pd.DataFrame) -> None:
    x_train, x_test, y_train, y_test = split_train_test(
        churn_df, target="Exited", test_size=0.25, random_state=0
    )

    assert "Exited" not in x_train.columns
    assert len(x_test) == pytest.approx(len(churn_df) * 0.25, abs=1)
    assert y_train.mean() == pytest.approx(y_test.mean(), abs=0.05)
