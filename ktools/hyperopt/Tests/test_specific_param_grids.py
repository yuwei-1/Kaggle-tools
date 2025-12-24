from functools import reduce
import unittest
from parameterized import parameterized
from sklearn.model_selection import KFold
from sklearn.metrics import root_mean_squared_error
from ktools.models import CatBoostModel
from ktools.models import LGBMModel
from ktools.models import XGBoostModel
from ktools.hyperopt.model_param_grids import *
from ktools.hyperopt.optuna_hyperparameter_optimizer import (
    OptunaHyperparameterOptimizer,
)
from ktools.preprocessing.basic_feature_transformers import *
from ktools.utils.data_science_pipeline_settings import DataSciencePipelineSettings


class TestSpecificParamGrids(unittest.TestCase):
    def setUp(self):
        train_csv_path = "data/used_car_prices/train.csv"
        test_csv_path = "data/used_car_prices/test.csv"
        target_col_name = "price"

        settings = DataSciencePipelineSettings(
            train_csv_path, test_csv_path, target_col_name
        )

        basic_transforms = [
            ConvertToLower.transform,
            FillNullValues.transform,
            ConvertObjectToCategorical.transform,
            LogTransformTarget.transform,
        ]

        self.basic_settings = reduce(
            lambda acc, func: func(acc), basic_transforms, settings
        )
        self.train_df, self.test_df = self.basic_settings.update()

        self.kf = KFold(n_splits=2, shuffle=True, random_state=123)

    @parameterized.expand(
        [(LGBMGBDTParamGrid()), (LGBMDARTParamGrid()), (LGBMRFParamGrid())]
    )
    def test_lgbm(self, lgbm_param_getter):
        model = LGBMModel

        optimizer = OptunaHyperparameterOptimizer(
            self.train_df.drop(columns=["price", "log_price"]),
            self.train_df["log_price"],
            model,
            lgbm_param_getter,
            self.kf,
            lambda y, yh: root_mean_squared_error(np.exp(y) - 1, np.exp(yh) - 1),
            direction="minimize",
            n_trials=2,
        )
        best_params = optimizer.optimize()

    @parameterized.expand([(XGBoostGBTree()), (XGBoostGBTreeLinear())])
    def test_xgb(self, xgb_param_getter):
        model = XGBoostModel

        optimizer = OptunaHyperparameterOptimizer(
            self.train_df.drop(columns=["price", "log_price"]),
            self.train_df["log_price"],
            model,
            xgb_param_getter,
            self.kf,
            lambda y, yh: root_mean_squared_error(np.exp(y) - 1, np.exp(yh) - 1),
            direction="minimize",
            n_trials=2,
        )
        best_params = optimizer.optimize()

    def test_catboost(self):
        model = CatBoostModel

        optimizer = OptunaHyperparameterOptimizer(
            self.train_df.drop(columns=["price", "log_price"]),
            self.train_df["log_price"],
            model,
            BaseCatBoostParamGrid(),
            self.kf,
            lambda y, yh: root_mean_squared_error(np.exp(y) - 1, np.exp(yh) - 1),
            direction="minimize",
            n_trials=2,
        )
        best_params = optimizer.optimize()
