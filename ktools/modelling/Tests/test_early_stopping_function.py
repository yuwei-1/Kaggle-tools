from functools import reduce
import unittest
import pandas as pd
from parameterized import parameterized
from sklearn.model_selection import train_test_split
from ktools.modelling.Interfaces.i_ktools_model import IKtoolsModel
from ktools.modelling.ktools_models.catboost_model import CatBoostModel
from ktools.modelling.ktools_models.lgbm_model import LGBMModel
from ktools.modelling.ktools_models.xgb_model import XGBoostModel
from ktools.preprocessing.basic_feature_transformers import *
from ktools.utils.data_science_pipeline_settings import DataSciencePipelineSettings


class TestKtoolsRegressionUsingDataset(unittest.TestCase):
    
    def setUp(self) -> None:
        train_csv_path = "/Users/yuwei-1/Documents/projects/Kaggle-tools/data/insurance/train.csv"
        test_csv_path = "/Users/yuwei-1/Documents/projects/Kaggle-tools/data/insurance/test.csv"
        self.target_col_name = "Premium Amount"

        settings = DataSciencePipelineSettings(
            train_csv_path,
            test_csv_path,
            self.target_col_name,
            train_data = pd.read_csv(train_csv_path).sample(100, random_state=42).reset_index(drop=True)
            )
                
        transforms = [
            FillNullValues.transform,
            OrdinalEncode.transform,
            ConvertObjectToCategorical.transform
        ]
        self.settings = settings = reduce(lambda acc, func: func(acc), transforms, settings)
        X, y, X_test, _ = settings.get_data()
        self.data_sets = train_test_split(X, y, random_state=42, shuffle=True, test_size=0.2)
    
    @parameterized.expand([
        (XGBoostModel(**{"objective" : "reg:squarederror", "num_boost_round" : 1000})),
        (CatBoostModel(**{"loss_function" : "RMSE", "num_boost_round" : 1000})),
        (LGBMModel(**{'objective': 'regression', "num_boost_round" : 1000}))
    ])
    def test_regression_early_stop(self, model_cls : IKtoolsModel):
        X_train, X_valid, y_train, y_valid = self.data_sets
        model_cls = model_cls.fit(X_train, y_train)
        y_pred = model_cls.predict(X_valid)
        num_trained_trees = model_cls.num_fitted_models

        self.assertTrue(y_pred.shape == y_valid.shape)
        self.assertTrue(num_trained_trees < 1000)

    @parameterized.expand([
        (XGBoostModel(**{"objective" : "reg:squarederror", "num_boost_round" : 1000, "early_stopping_rounds" : None})),
        (CatBoostModel(**{"loss_function" : "RMSE", "num_boost_round" : 1000, "early_stopping_rounds" : None})),
        (LGBMModel(**{'objective': 'regression', "num_boost_round" : 1000, "early_stopping_rounds" : None}))
    ])
    def test_regression(self, model_cls : IKtoolsModel):
        X_train, X_valid, y_train, y_valid = self.data_sets
        model_cls = model_cls.fit(X_train, y_train)
        y_pred = model_cls.predict(X_valid)
        num_trained_trees = model_cls.num_fitted_models

        self.assertTrue(y_pred.shape == y_valid.shape)
        self.assertTrue(num_trained_trees == 1000)