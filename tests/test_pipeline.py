import pandas as pd
import pytest

from churn.config import Config
from churn.pipeline import MODEL_NAMES, build_pipeline


@pytest.mark.parametrize("model_name", MODEL_NAMES)
def test_every_model_fits_and_predicts(
    model_name: str, churn_df: pd.DataFrame, config: Config
) -> None:
    pipe = build_pipeline(model_name, config.features, random_state=0)
    if model_name == "catboost":
        pipe.set_params(classifier__iterations=10)
    x, y = churn_df.drop(columns="Exited"), churn_df["Exited"]

    pipe.fit(x, y)
    proba = pipe.predict_proba(x)

    assert proba.shape == (len(x), 2)


def test_low_importance_features_are_dropped(churn_df: pd.DataFrame, config: Config) -> None:
    pipe = build_pipeline("decision_tree", config.features, drop_low_importance=True)
    x, y = churn_df.drop(columns="Exited"), churn_df["Exited"]

    pipe.fit(x, y)
    names = pipe.named_steps["preprocessing"].get_feature_names_out()

    for column in config.features.low_importance:
        assert not any(column in name for name in names)
    assert any("HasBalance" in name for name in names)


def test_unknown_model_raises(config: Config) -> None:
    with pytest.raises(ValueError, match="Unknown model"):
        build_pipeline("svm", config.features)
