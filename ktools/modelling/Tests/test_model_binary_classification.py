from typing import Any
import unittest
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from ktools.modelling.models.catboost_model import CatBoostModel
from ktools.modelling.models.lgbm_model import LGBMModel
from ktools.modelling.models.pytorch_ffn_model import PytorchFFNModel
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

    def test_pytorch_nn(self):
        train_features = list(self.train_df.columns)[:-1]
        oe = OrdinalEncoder()
        self.train_df[train_features] = oe.fit_transform(self.train_df[train_features])
        cat_sizes = [int(x) for x in self.train_df[train_features].max().values]
        cat_emb = [int(np.sqrt(x)) for x in cat_sizes]
        pynn = PytorchFFNModel(len(train_features),
                               output_dim=1,
                               categorical_idcs=list(range(len(train_features))),
                               categorical_sizes=cat_sizes,
                               categorical_embedding=cat_emb,
                               activation='gelu',
                               last_activation='sigmoid',
                               loss=nn.BCELoss())
        
        pynn.fit(self.train_df.drop(columns=self.target_col_name), self.train_df[[self.target_col_name]])

        print()