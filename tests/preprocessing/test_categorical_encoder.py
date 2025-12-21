import numpy as np
import pandas as pd
import pytest
from ktools.config.dataset import DatasetConfig
from ktools.preprocessing.categorical import CategoricalEncoder


EXPECTED_TRAIN_ENCODED_VALUES = np.array(
    [[0, 0], [1, 1], [0, 0], [2, 1], [1, -1], [0, 0], [3, 2], [2, 1], [1, 0], [0, 2]],
    dtype=int,
)

EXPECTED_VAL_ENCODED_VALUES = np.array(
    [[1, 1], [2, 2], [3, 0], [0, 1], [-2, -2], [-1, -1]], dtype=int
)

CONFIG = DatasetConfig(
    training_col_names=["category_1", "category_2"],
    numerical_col_names=[],
    categorical_col_names=["category_1", "category_2"],
    target_col_name="",
)


@pytest.fixture
def dummy_data() -> pd.DataFrame:
    data = pd.DataFrame(
        {
            "category_1": ["A", "B", "A", "C", "B", "A", "D", "C", "B", "A"],
            "category_2": ["X", "Y", "X", "Y", np.nan, "X", "Z", "Y", "X", "Z"],
        }
    )

    val_data = pd.DataFrame(
        {
            "category_1": ["B", "C", "D", "A", "E", np.nan],
            "category_2": ["Y", "Z", "X", "Y", "W", np.nan],
        }
    )
    data = pd.concat([data, val_data], keys=["train", "val"])
    return data


def test_categorical_encoder(dummy_data: pd.DataFrame):
    train_data, val_data = dummy_data.loc["train"], dummy_data.loc["val"]
    encoder = CategoricalEncoder(CONFIG)
    train_data_encoded = encoder.fit_transform(train_data)
    val_data_encoded = encoder.transform(val_data)

    assert encoder.fitted, "encoder should be marked as fitted after fit_transform"
    np.testing.assert_array_equal(
        train_data_encoded.to_numpy(),
        EXPECTED_TRAIN_ENCODED_VALUES,
        err_msg="encoded training data does not match expected values",
    )

    np.testing.assert_array_equal(
        val_data_encoded.to_numpy(),
        EXPECTED_VAL_ENCODED_VALUES,
        err_msg="encoded validation data does not match expected values",
    )

    assert train_data_encoded.dtypes.tolist() == [np.int64, np.int64], (
        "expected encoded training data to be of integer dtype"
    )


def test_categorical_encoder_load_save(tmp_path, dummy_data: pd.DataFrame):
    train_data, val_data = dummy_data.loc["train"], dummy_data.loc["val"]
    encoder = CategoricalEncoder(CONFIG)
    encoder.fit(train_data)
    encoder.save(tmp_path)

    loaded_encoder = CategoricalEncoder.load(tmp_path)

    train_data_encoded = loaded_encoder.transform(train_data)
    val_data_encoded = loaded_encoder.transform(val_data)

    np.testing.assert_array_equal(
        train_data_encoded.to_numpy(),
        EXPECTED_TRAIN_ENCODED_VALUES,
        err_msg="encoded training data from loaded encoder does not match expected values",
    )

    np.testing.assert_array_equal(
        val_data_encoded.to_numpy(),
        EXPECTED_VAL_ENCODED_VALUES,
        err_msg="encoded validation data from loaded encoder does not match expected values",
    )
