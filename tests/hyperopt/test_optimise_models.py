from typing import Type
import pandas as pd
import pytest
from sklearn.datasets import make_regression
from sklearn.model_selection import KFold
from sklearn.metrics import root_mean_squared_error
from ktools.base.model import BaseKtoolsModel
from ktools.config.dataset import DatasetConfig
from ktools.fitting.pipe import ModelPipeline
from ktools.hyperopt.optuna_optimiser import OptunaHyperparameterOptimizer
from ktools.models import LGBMModel, CatBoostModel, XGBoostModel
from ktools.preprocessing.pipe import PreprocessingPipeline


FEATURES = [f"feature_{i}" for i in range(10)]
TARGET = "target"

CONFIG = DatasetConfig(
    training_col_names=FEATURES,
    target_col_name=TARGET,
    numerical_col_names=FEATURES,
    categorical_col_names=[],
)

TUNING_CONFIGS = [
    ("./tests/hyperopt/grids/test_lgbm.yml", "gbdt-bagging", LGBMModel),
    ("./tests/hyperopt/grids/test_lgbm.yml", "gbdt-goss", LGBMModel),
    ("./tests/hyperopt/grids/test_lgbm.yml", "dart", LGBMModel),
    ("./tests/hyperopt/grids/test_lgbm.yml", "rf", LGBMModel),
    ("./tests/hyperopt/grids/test_xgboost.yml", "base", XGBoostModel),
    ("./tests/hyperopt/grids/test_xgboost.yml", "gbtree", XGBoostModel),
    ("./tests/hyperopt/grids/test_xgboost.yml", "gblinear", XGBoostModel),
    ("./tests/hyperopt/grids/test_xgboost.yml", "dart", XGBoostModel),
    ("./tests/hyperopt/grids/test_catboost.yml", "base", CatBoostModel),
    ("./tests/hyperopt/grids/test_catboost.yml", "mvs", CatBoostModel),
    ("./tests/hyperopt/grids/test_catboost.yml", "bayesian", CatBoostModel),
    ("./tests/hyperopt/grids/test_catboost.yml", "bernoulli", CatBoostModel),
]


@pytest.fixture
def dummy_data() -> pd.DataFrame:
    X, y = make_regression(
        n_samples=10000,
        n_features=10,
        n_informative=5,
        noise=0.1,
        random_state=42,
    )

    df = pd.DataFrame(X, columns=FEATURES)
    df[TARGET] = y
    return df


def cv_tunable_func(
    train_data: pd.DataFrame,
    model_class: Type[BaseKtoolsModel],
    **model_params,
) -> float:
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    mean_score: float = 0.0
    for train_index, val_index in kfold.split(
        train_data, train_data[CONFIG.target_col_name]
    ):
        train_fold: pd.DataFrame = train_data.iloc[train_index].copy()
        val_fold: pd.DataFrame = train_data.iloc[val_index]

        preprocessor_pipeline = PreprocessingPipeline(preprocessors=[])

        pipe = ModelPipeline(
            model=model_class(**model_params),
            config=CONFIG,
            preprocessor=preprocessor_pipeline,
        )

        pipe.fit(train_fold, validation_data=val_fold)
        y_pred = pipe.predict(val_fold)
        score = root_mean_squared_error(val_fold[CONFIG.target_col_name], y_pred)
        mean_score += score / 5

    return mean_score


@pytest.mark.parametrize("yml_path,model_type,model_class", TUNING_CONFIGS)
def test_tune_model_hyperparameters(
    yml_path: str, model_type: str, model_class: Type[BaseKtoolsModel], dummy_data
):
    X = dummy_data

    optimiser = OptunaHyperparameterOptimizer(
        model_type=model_type,
        grid_yaml_path=yml_path,
        n_trials=10,
    )

    best_params = optimiser.optimize(
        X,
        model_class,
        tunable_func=cv_tunable_func,
    )

    assert isinstance(best_params, dict)
    # assert len(best_params) > 0
