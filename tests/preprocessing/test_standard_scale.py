import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_regression
from ktools.config.dataset import DatasetConfig
from ktools.preprocessing.numerical import StandardScale


EXPECTED_TRAIN_SCALED_VALUES = [-2, -1, 0, 0, 0, 0, 0, 0, 1, 2]
EXPECTED_VAL_SCALED_VALUES = [0, 1, -1, -2, 2]

CONFIG = DatasetConfig(
    training_col_names=["feature_0"],
    numerical_col_names=["feature_0"],
    categorical_col_names=[],
    target_col_name="",
)

NUM_FEATURES = 5
TRAINING_COL_NAMES = [f"feature_{i}" for i in range(NUM_FEATURES)]


@pytest.fixture
def dummy_data() -> pd.DataFrame:
    X, _ = make_regression(n_samples=100, n_features=NUM_FEATURES, random_state=42)
    df = pd.DataFrame(X, columns=TRAINING_COL_NAMES)
    return df


@pytest.fixture
def simple_data() -> pd.DataFrame:
    data = pd.DataFrame({"feature_0": [1, 2, 3, 3, 3, 3, 3, 3, 4, 5]})

    val_data = pd.DataFrame({"feature_0": [3, 4, 2, 1, 5]})
    df = pd.concat([data, val_data], keys=["train", "val"])
    return df


def test_standard_scale(dummy_data):
    config = DatasetConfig(
        training_col_names=TRAINING_COL_NAMES,
        numerical_col_names=TRAINING_COL_NAMES,
        categorical_col_names=[],
        target_col_name="",
    )

    scaler = StandardScale(config)
    dummy_data_scaled = scaler.fit_transform(dummy_data)

    # Using ddof = 0 as sklearn's StandardScaler uses population std deviation
    assert scaler.fitted, "scaler should be marked as fitted after fit_transform"
    assert np.allclose(dummy_data_scaled.mean().values, 0, atol=1e-7), (
        "expected mean to be close to 0 after standard scaling"
    )
    assert np.allclose(dummy_data_scaled.std(ddof=0).values, 1, atol=1e-7), (
        "expected std to be close to 1 after standard scaling"
    )


def test_standard_scale_simple(simple_data: pd.DataFrame):
    train_data, val_data = simple_data.loc["train"], simple_data.loc["val"]
    scaler = StandardScale(CONFIG)
    train_data_scaled = scaler.fit_transform(train_data)
    val_data_scaled = scaler.transform(val_data)

    assert (
        train_data_scaled["feature_0"].values.tolist() == EXPECTED_TRAIN_SCALED_VALUES
    ), "scaled values do not match expected values"
    assert val_data_scaled["feature_0"].values.tolist() == EXPECTED_VAL_SCALED_VALUES, (
        "scaled values do not match expected values"
    )


def test_standard_scale_save_load(tmp_path, simple_data: pd.DataFrame):
    train_data, val_data = simple_data.loc["train"], simple_data.loc["val"]
    scaler = StandardScale(CONFIG)
    scaler.fit(train_data)
    scaler.save(tmp_path)

    loaded_scaler = StandardScale.load(tmp_path)
    val_data_scaled = loaded_scaler.transform(val_data)

    assert val_data_scaled["feature_0"].values.tolist() == EXPECTED_VAL_SCALED_VALUES, (
        "scaled values from loaded scaler do not match expected values"
    )
