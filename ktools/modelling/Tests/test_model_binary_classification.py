from typing import Any
import unittest

import numpy as np

from ktools.modelling.models.catboost_model import CatBoostModel
from ktools.modelling.models.lgbm_model import LGBMModel
from ktools.preprocessing.basic_feature_transformers import ConvertObjectToCategorical
from ktools.utils.data_science_pipeline_settings import DataSciencePipelineSettings


class TestModelBinaryClassification(unittest.TestCase):

    def setUp(self) -> None:
        train_csv_path = "/Users/yuwei-1/Documents/projects/Kaggle-tools/data/loan_approval/train.csv"
        test_csv_path = "/Users/yuwei-1/Documents/projects/Kaggle-tools/data/loan_approval/test.csv"
        self.target_col_name = "loan_status"

        settings = DataSciencePipelineSettings(train_csv_path,
                                               test_csv_path,
                                               self.target_col_name, 
                                               )
        settings = ConvertObjectToCategorical.transform(settings)
        self.train_df, self.test_df = settings.update()
        self.test_df.drop(columns=[self.target_col_name], inplace=True)


    def test_lgbm(self):
        lgbm = LGBMModel(**{'objective': 'binary', 'metric': 'binary_logloss'})
        lgbm.fit(self.train_df.drop(columns=self.target_col_name), self.train_df[self.target_col_name])
        y_pred = lgbm.predict(self.test_df)

        self.assertTrue((y_pred >= 0).all())
        self.assertTrue((y_pred <= 1).all())
        self.assertTrue(len(np.unique(y_pred)) > 2)

    def test_cat(self):
        cat = CatBoostModel(**{'loss_function':'Logloss', 'eval_metric':"AUC"})
        cat.fit(self.train_df.drop(columns=self.target_col_name), self.train_df[self.target_col_name])
        y_pred = cat.predict(self.test_df)

        self.assertTrue((y_pred >= 0).all())
        self.assertTrue((y_pred <= 1).all())
        self.assertTrue(len(np.unique(y_pred)) > 2)