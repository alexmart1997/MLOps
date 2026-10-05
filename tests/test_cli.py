from dataclasses import asdict
from pathlib import Path

import pandas as pd
import yaml

from churn.cli import main
from churn.config import Config


def test_cli_train_then_predict(tmp_path: Path, config: Config, churn_df: pd.DataFrame) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(asdict(config)), encoding="utf-8")
    input_path = tmp_path / "new.csv"
    churn_df.drop(columns="Exited").head(10).to_csv(input_path, index=False)
    output_path = tmp_path / "pred.csv"

    main(["-c", str(config_path), "train"])
    main(["-c", str(config_path), "predict", str(input_path), "-o", str(output_path)])

    assert len(pd.read_csv(output_path)) == 10
