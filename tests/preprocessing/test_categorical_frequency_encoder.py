import numpy as np
import pandas as pd
import pytest
from ktools.config.dataset import DatasetConfig
from ktools.preprocessing.categorical import CategoricalFrequencyEncoder


CONFIG = DatasetConfig(
    training_col_names=["category_1", "category_2"],
    numerical_col_names=[],
    categorical_col_names=["category_1", "category_2"],
    target_col_name="",
)


@pytest.fixture
def dummy_data() -> pd.DataFrame:
    train_data = pd.DataFrame(
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
    data = pd.concat([train_data, val_data], keys=["train", "val"])
    return data


EXPECTED_TRAIN_FREQ_VALUES = np.array(
    [
        [0.4, 4 / 9],
        [0.3, 3 / 9],
        [0.4, 4 / 9],
        [0.2, 3 / 9],
        [0.3, 0.0],  # NaN in category_2 filled with 0
        [0.4, 4 / 9],
        [0.1, 2 / 9],
        [0.2, 3 / 9],
        [0.3, 4 / 9],
        [0.4, 2 / 9],
    ],
    dtype=float,
)

EXPECTED_VAL_FREQ_VALUES = np.array(
    [
        [0.3, 3 / 9],
        [0.2, 2 / 9],
        [0.1, 4 / 9],
        [0.4, 3 / 9],
        [0.0, 0.0],  # E and W are unseen categories
        [0.0, 0.0],  # NaN in both columns filled with 0
    ],
    dtype=float,
)


def test_categorical_frequency_encoder(dummy_data: pd.DataFrame):
    train_data, val_data = dummy_data.loc["train"], dummy_data.loc["val"]

    encoder = CategoricalFrequencyEncoder(CONFIG)
    train_data_encoded = encoder.fit_transform(train_data)
    val_data_encoded = encoder.transform(val_data)

    assert encoder.fitted, "encoder should be marked as fitted after fit_transform"

    assert "category_1" in train_data_encoded.columns
    assert "category_2" in train_data_encoded.columns

    freq_col_1 = "category_1" + CategoricalFrequencyEncoder.freq_suffix
    freq_col_2 = "category_2" + CategoricalFrequencyEncoder.freq_suffix
    assert freq_col_1 in train_data_encoded.columns
    assert freq_col_2 in train_data_encoded.columns

    np.testing.assert_array_almost_equal(
        train_data_encoded[[freq_col_1, freq_col_2]].to_numpy(),
        EXPECTED_TRAIN_FREQ_VALUES,
        decimal=5,
        err_msg="encoded training frequency data does not match expected values",
    )

    np.testing.assert_array_almost_equal(
        val_data_encoded[[freq_col_1, freq_col_2]].to_numpy(),
        EXPECTED_VAL_FREQ_VALUES,
        decimal=5,
        err_msg="encoded validation frequency data does not match expected values",
    )

    assert train_data_encoded[freq_col_1].dtype == float
    assert train_data_encoded[freq_col_2].dtype == float


def test_categorical_frequency_encoder_load_save(tmp_path, dummy_data: pd.DataFrame):
    train_data, val_data = dummy_data.loc["train"], dummy_data.loc["val"]

    encoder = CategoricalFrequencyEncoder(CONFIG)
    encoder.fit(train_data)
    encoder.save(tmp_path)

    loaded_encoder = CategoricalFrequencyEncoder.load(tmp_path)

    train_data_encoded = loaded_encoder.transform(train_data)
    val_data_encoded = loaded_encoder.transform(val_data)

    freq_col_1 = "category_1" + CategoricalFrequencyEncoder.freq_suffix
    freq_col_2 = "category_2" + CategoricalFrequencyEncoder.freq_suffix

    np.testing.assert_array_almost_equal(
        train_data_encoded[[freq_col_1, freq_col_2]].to_numpy(),
        EXPECTED_TRAIN_FREQ_VALUES,
        decimal=5,
        err_msg="encoded training data from loaded encoder does not match expected values",
    )

    np.testing.assert_array_almost_equal(
        val_data_encoded[[freq_col_1, freq_col_2]].to_numpy(),
        EXPECTED_VAL_FREQ_VALUES,
        decimal=5,
        err_msg="encoded validation data from loaded encoder does not match expected values",
    )
