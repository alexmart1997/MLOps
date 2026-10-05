"""Сборка sklearn-пайплайнов для моделей-кандидатов."""

from __future__ import annotations

from collections.abc import Callable

from catboost import CatBoostClassifier
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    OneHotEncoder,
    PolynomialFeatures,
    StandardScaler,
)
from sklearn.tree import DecisionTreeClassifier

from churn.config import FeaturesConfig
from churn.features import add_has_balance, drop_columns

_CLASSIFIERS: dict[str, Callable[[int], BaseEstimator]] = {
    "logreg": lambda seed: LogisticRegression(random_state=seed, max_iter=1000),
    "decision_tree": lambda seed: DecisionTreeClassifier(random_state=seed),
    "random_forest": lambda seed: RandomForestClassifier(random_state=seed),
    "catboost": lambda seed: CatBoostClassifier(
        random_state=seed, verbose=0, allow_writing_files=False
    ),
}
MODEL_NAMES = tuple(_CLASSIFIERS)

# Линейным моделям нужны полиномиальные признаки и скейлинг, деревьям — нет.
_LINEAR_MODELS = {"logreg"}


def _one_hot() -> OneHotEncoder:
    return OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")


def build_pipeline(
    model_name: str,
    features: FeaturesConfig,
    random_state: int = 42,
    drop_low_importance: bool = False,
) -> Pipeline:
    """Пайплайн «сырой датафрейм → предсказание» для указанной модели.

    Все преобразования (удаление колонок, feature engineering, кодирование)
    находятся внутри пайплайна, поэтому сохранённая модель принимает
    данные в исходном формате.
    """
    if model_name not in _CLASSIFIERS:
        raise ValueError(f"Unknown model {model_name!r}; expected one of {MODEL_NAMES}")

    to_drop = list(features.drop)
    if drop_low_importance:
        to_drop += features.low_importance
    numeric = [col for col in features.numeric if col not in to_drop]
    categorical = [col for col in features.categorical if col not in to_drop]

    transformers: list[tuple[str, object, list[str]]] = [("cat", _one_hot(), categorical)]
    if model_name in _LINEAR_MODELS:
        transformers.insert(0, ("num", SimpleImputer(strategy="median"), numeric))
    preprocessor = ColumnTransformer(
        transformers, remainder="passthrough", verbose_feature_names_out=True
    )

    steps: list[tuple[str, object]] = [
        ("drop_cols", FunctionTransformer(drop_columns, kw_args={"columns": to_drop})),
        ("add_has_balance", FunctionTransformer(add_has_balance)),
        ("preprocessing", preprocessor),
    ]
    if model_name in _LINEAR_MODELS:
        steps += [
            ("poly_features", PolynomialFeatures(degree=2, include_bias=False)),
            ("final_scaler", StandardScaler()),
        ]
    steps.append(("classifier", _CLASSIFIERS[model_name](random_state)))
    return Pipeline(steps)
