import pandas as pd

from churn.features import add_has_balance, drop_columns


def test_drop_columns_ignores_absent(churn_df: pd.DataFrame) -> None:
    result = drop_columns(churn_df, ["id", "NotAColumn"])

    assert "id" not in result.columns
    assert "id" in churn_df.columns, "input must not be mutated"


def test_add_has_balance() -> None:
    df = pd.DataFrame({"Balance": [0.0, 10.5, 0.0]})

    result = add_has_balance(df)

    assert result["HasBalance"].tolist() == [0, 1, 0]
    assert "HasBalance" not in df.columns, "input must not be mutated"
