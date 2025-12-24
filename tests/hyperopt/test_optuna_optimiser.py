import pytest
from ktools.hyperopt.optuna_optimiser import OptunaHyperparameterOptimizer
from pathlib import Path
from typing import Tuple
import pandas as pd
from ktools.models import LGBMModel
from ktools.preprocessing.categorical import CategoricalEncoder
from ktools.preprocessing.numerical import StandardScale
from ktools.preprocessing.pipe import PreprocessingPipeline
from ktools.config.dataset import DatasetConfig
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score


"""
Integration test for OptunaHyperparameterOptimizer class
"""

EXPECTED_PARAM_GRID = {
    "num_leaves": 8,
    "max_depth": 3,
    "learning_rate": 0.49645362087103995,
    "num_boost_round": 5,
    "subsample": 0.9693992246777331,
    "colsample_bytree": 0.8089456215759775,
    "reg_alpha": 0.9993688142551358,
    "reg_lambda": 0.0195668091150637,
    "min_data_in_leaf": 5,
    "bagging_freq": 1,
    "min_child_weight": 9.9233980685643,
    "cat_smooth": 9.967694844912314,
}
EXPECTED_BEST_VALUES = 0.6911822444509161
DATA_PATH = Path("./data/diabetes_prediction/")
TARGET = "diagnosed_diabetes"
GRID_PATH = Path("./tests/hyperopt/grids/lgbm_fast.yml")
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


def test_optuna_optimizer(diabetes_data: OUT_TYPE):
    config, train_data, test_data = diabetes_data

    preprocessors = [StandardScale(config), CategoricalEncoder(config)]
    preprocessor = PreprocessingPipeline(preprocessors=preprocessors)

    optimizer = OptunaHyperparameterOptimizer(
        model=LGBMModel,
        grid_yaml_path=str(GRID_PATH),
        config=config,
        evaluation_metric=roc_auc_score,
        kfold_object=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
        preprocessor=preprocessor,
        timeout=60,
        model_type="base",
        direction="maximize",
        n_trials=3,
        study_name="test_optuna_optimizer",
        explore_fraction=0.1,
        save_study=False,
        verbose=False,
        random_state=42,
    )

    optimal_params = optimizer.optimize(
        train_data=train_data,
        test_data=test_data,
    )

    assert optimal_params == EXPECTED_PARAM_GRID, (
        f"Optimal params do not match expected. Got: {optimal_params}"
    )
    assert optimizer.study.best_value == EXPECTED_BEST_VALUES, (
        f"Best value does not match expected. Got: {optimizer.study.best_value}"
    )

    # Assert that expected hyperparameters are present
    expected_params = [
        "num_leaves",
        "max_depth",
        "learning_rate",
        "num_boost_round",
        "subsample",
        "colsample_bytree",
        "reg_alpha",
        "reg_lambda",
        "min_data_in_leaf",
        "bagging_freq",
        "min_child_weight",
        "cat_smooth",
    ]
    for param in expected_params:
        assert param in optimal_params, (
            f"Expected parameter '{param}' not found in optimal_params"
        )

    # Assert that the study has best trial
    assert optimizer.study.best_trial is not None
    assert optimizer.study.best_value > 0
