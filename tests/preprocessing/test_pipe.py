import numpy as np
import pandas as pd
import pytest
from ktools.config.dataset import DatasetConfig
from ktools.preprocessing.categorical import CategoricalEncoder
from ktools.preprocessing.numerical import StandardScale
from ktools.preprocessing.pipe import PreprocessingPipeline


CONFIG = DatasetConfig(
    training_col_names=["feature_0", "category_1"],
    numerical_col_names=["feature_0"],
    categorical_col_names=["category_1"],
    target_col_name="",
)

EXPECTED_TRAIN_PROCESSED_VALUES = pd.DataFrame(
    {
        "feature_0": [-2, -1, 0, 0, 0, 0, 0, 0, 1, 2],
        "category_1": [0, 1, 0, 2, 1, 0, 3, 2, 1, 0],
    }
).astype({"category_1": "category", "feature_0": "float64"})

EXPECTED_VAL_PROCESSED_VALUES = pd.DataFrame(
    {"feature_0": [0, 1, -1, -2, 2, 2], "category_1": [1, 2, 3, 0, -2, -1]},
    # dtype={"category_1": "category", "feature_0": "float64"},
).astype({"category_1": "category", "feature_0": "float64"})


@pytest.fixture
def dummy_data() -> pd.DataFrame:
    data = pd.DataFrame(
        {
            "feature_0": [1, 2, 3, 3, 3, 3, 3, 3, 4, 5],
            "category_1": ["A", "B", "A", "C", "B", "A", "D", "C", "B", "A"],
        }
    )

    val_data = pd.DataFrame(
        {
            "feature_0": [3, 4, 2, 1, 5, 5],
            "category_1": ["B", "C", "D", "A", "E", np.nan],
        }
    )

    data = pd.concat([data, val_data], keys=["train", "val"])
    return data


def test_preprocessing_pipeline(dummy_data: pd.DataFrame):
    train_data, val_data = dummy_data.loc["train"], dummy_data.loc["val"]

    preprocessors = [StandardScale(CONFIG), CategoricalEncoder(CONFIG)]

    pipeline = PreprocessingPipeline(preprocessors=preprocessors)

    train_data_processed = pipeline.train_pipe(train_data)
    val_data_processed = pipeline.inference_pipe(val_data)

    pd.testing.assert_frame_equal(train_data_processed, EXPECTED_TRAIN_PROCESSED_VALUES)
    pd.testing.assert_frame_equal(val_data_processed, EXPECTED_VAL_PROCESSED_VALUES)


def test_preprocessing_pipeline_save_load(tmp_path, dummy_data: pd.DataFrame):
    train_data, val_data = dummy_data.loc["train"], dummy_data.loc["val"]

    preprocessors = [StandardScale(CONFIG), CategoricalEncoder(CONFIG)]

    pipeline = PreprocessingPipeline(preprocessors=preprocessors)
    pipeline.train_pipe(train_data)
    pipeline.save(tmp_path)

    loaded_pipeline = PreprocessingPipeline.load(tmp_path)
    val_data_processed = loaded_pipeline.inference_pipe(val_data)

    pd.testing.assert_frame_equal(val_data_processed, EXPECTED_VAL_PROCESSED_VALUES)
