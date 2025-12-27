from pathlib import Path
import numpy as np
import pandas as pd
from ktools.utils.load_dataframes import load_all_dataframes


PRED_SIZE = 100


def test_load_dataframes(tmp_path: Path):
    for i in range(10):
        df = pd.DataFrame(
            index=range(PRED_SIZE),
            data={
                f"col_{i}": np.full(PRED_SIZE, i),
            },
        )

        df.to_csv(tmp_path / f"file_{i}.csv", index=False)

    all_predictions = load_all_dataframes(dir_path=tmp_path.as_posix())

    assert all_predictions.shape == (PRED_SIZE, 10)
    assert (all_predictions.iloc[0].to_numpy() == np.arange(10)).all()
