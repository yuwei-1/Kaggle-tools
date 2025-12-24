import pytest
from ktools.fitting.cv_executor import CrossValidationExecutor
from pathlib import Path
from typing import Tuple
import pandas as pd
from ktools.models import LGBMModel
from ktools.preprocessing.categorical import CategoricalEncoder
from ktools.preprocessing.numerical import StandardScale
from ktools.preprocessing.pipe import PreprocessingPipeline
from ktools.config.dataset import DatasetConfig
from ktools.fitting.pipe import ModelPipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score


"""
Integration test for model pipeline class
"""


DATA_PATH = Path("./data/diabetes_prediction/")
TARGET = "diagnosed_diabetes"
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


def test_integration_cv(diabetes_data: OUT_TYPE):
    config, train_data, test_data = diabetes_data

    preprocessors = [StandardScale(config), CategoricalEncoder(config)]

    pipe = ModelPipeline(
        model=LGBMModel(),
        config=config,
        preprocessor=PreprocessingPipeline(preprocessors=preprocessors),
    )

    cv_executor = CrossValidationExecutor(
        config=config,
        model_pipeline=pipe,
        evaluation_metric=roc_auc_score,
        kfold_object=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    )

    mean_score, train_oof_preds, _, test_oof_preds = cv_executor.run(
        train_data=train_data, test_data=test_data
    )

    assert train_oof_preds.shape[0] == train_data.shape[0]
    assert test_oof_preds.shape[0] == test_data.shape[0]
    assert mean_score > 0.72
