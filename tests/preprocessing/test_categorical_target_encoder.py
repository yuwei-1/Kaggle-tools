import numpy as np
import pandas as pd
import pytest
from ktools.config.dataset import DatasetConfig
from ktools.preprocessing.categorical import CategoricalTargetEncoder


CONFIG = DatasetConfig(
    training_col_names=["category_1", "category_2"],
    numerical_col_names=[],
    categorical_col_names=["category_1", "category_2"],
    target_col_name="target",
)


@pytest.fixture
def dummy_data() -> pd.DataFrame:
    train_data = pd.DataFrame(
        {
            "category_1": ["A", "B", "A", "C", "B", "A", "C", "C", "B", "A"],
            "category_2": ["X", "Y", "X", "Y", "X", "X", "Y", "Y", "X", "Y"],
            "target": [1, 0, 1, 0, 1, 1, 0, 1, 0, 1],
        }
    )

    val_data = pd.DataFrame(
        {
            "category_1": ["A", "B", "C"],
            "category_2": ["X", "Y", "X"],
            "target": [1, 0, 1],
        }
    )
    data = pd.concat([train_data, val_data], keys=["train", "val"])
    return data


def test_categorical_target_encoder(dummy_data: pd.DataFrame):
    train_data, val_data = dummy_data.loc["train"], dummy_data.loc["val"]

    encoder = CategoricalTargetEncoder(CONFIG, random_state=42, cv=2, smooth=1)
    train_data_encoded = encoder.fit_transform(train_data)
    val_data_encoded = encoder.transform(val_data)

    assert encoder.fitted, "encoder should be marked as fitted after fit_transform"

    assert train_data_encoded["category_1"].dtype == np.float32
    assert train_data_encoded["category_2"].dtype == np.float32
    assert val_data_encoded["category_1"].dtype == np.float32
    assert val_data_encoded["category_2"].dtype == np.float32

    assert "target" in train_data_encoded.columns
    assert "target" in val_data_encoded.columns

    assert train_data_encoded["category_1"].min() >= 0
    assert train_data_encoded["category_1"].max() <= 1
    assert train_data_encoded["category_2"].min() >= 0
    assert train_data_encoded["category_2"].max() <= 1

    val_cat1_values = val_data_encoded["category_1"].values
    assert val_cat1_values[0] > val_cat1_values[1], (
        "Category A should have higher target encoding than B"
    )
