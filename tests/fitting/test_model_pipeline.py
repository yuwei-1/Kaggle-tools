from pathlib import Path
from typing import Tuple
import pandas as pd
import pytest
from ktools.models import LGBMModel
from ktools.preprocessing.categorical import CategoricalEncoder
from ktools.preprocessing.numerical import StandardScale
from ktools.preprocessing.pipe import PreprocessingPipeline
from ktools.config.dataset import DatasetConfig
from ktools.fitting.pipe import ModelPipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score


"""
Integration test for model pipeline class
"""


DATA_PATH = Path("./data/diabetes_prediction/")
TARGET = "diagnosed_diabetes"
EXPECTED_ROC_AUC = 0.7221312966927544
OUT_TYPE = Tuple[DatasetConfig, pd.DataFrame, pd.DataFrame]


@pytest.fixture
def diabetes_data() -> OUT_TYPE:
    train_data = pd.read_csv(DATA_PATH / "train.csv", index_col=0)
    test_data = pd.read_csv(DATA_PATH / "test.csv", index_col=0)

    training_col_names = train_data.drop(columns=TARGET).columns.tolist()
    numerical_col_names = (
        train_data.drop(columns=TARGET)
        .select_dtypes(include=["number"])
        .columns.tolist()
    )
    categorical_col_names = train_data.select_dtypes(
        include=["object"]
    ).columns.tolist()

    config = DatasetConfig(
        training_col_names=training_col_names,
        numerical_col_names=numerical_col_names,
        categorical_col_names=categorical_col_names,
        target_col_name=TARGET,
    )

    return config, train_data, test_data


def test_integration_model_pipeline(diabetes_data: OUT_TYPE):
    config, train_data, _ = diabetes_data
    train_data, val_data = train_test_split(
        train_data, test_size=0.2, random_state=42, shuffle=True
    )

    preprocessors = [StandardScale(config), CategoricalEncoder(config)]

    pipe = ModelPipeline(
        model=LGBMModel(),
        config=config,
        preprocessor=PreprocessingPipeline(preprocessors=preprocessors),
    )

    pipe.fit(train_data)
    y_pred = pipe.predict(val_data.drop(columns=[TARGET]))

    score = roc_auc_score(val_data[TARGET], y_pred)
    assert score == EXPECTED_ROC_AUC, "Expected ROC AUC does not match actual"
