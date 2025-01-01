from typing import Any
import unittest
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from ktools.modelling.ktools_models.catboost_model import CatBoostModel
from ktools.modelling.ktools_models.hgb_model import HGBModel
from ktools.modelling.ktools_models.keras_embedding_model import KerasEmbeddingModel
from ktools.modelling.ktools_models.keras_factorization_machine import KerasFM
from ktools.modelling.ktools_models.lgbm_model import LGBMModel
from ktools.modelling.ktools_models.pytorch_ffn_model import PytorchFFNModel
from ktools.modelling.ktools_models.tabnet_model import TabNetModel
from ktools.modelling.ktools_models.xgb_model import XGBoostModel
from ktools.modelling.ktools_models.yggdrasil_gbt_model import YDFGBoostModel
from ktools.preprocessing.basic_feature_transformers import ConvertObjectToCategorical, OrdinalEncode
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
    
    def test_ydf(self):
        yggdrasil = YDFGBoostModel(task="CLASSIFICATION", loss="BINOMIAL_LOG_LIKELIHOOD")

        self.train_df[self.target_col_name] = self.train_df[self.target_col_name].astype(int)
        yggdrasil.fit(self.train_df.drop(columns=self.target_col_name), self.train_df[self.target_col_name])
        y_pred = yggdrasil.predict(self.test_df)

        self.assertTrue((y_pred >= 0).all())
        self.assertTrue((y_pred <= 1).all())
        self.assertTrue(len(np.unique(y_pred)) > 2)

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

    def test_xgb(self):
        xgb = XGBoostModel(**{"objective": "binary:logistic", "eval_metric": "logloss"})
        xgb.fit(self.train_df.drop(columns=self.target_col_name), self.train_df[self.target_col_name])
        y_pred = xgb.predict(self.test_df)

        self.assertTrue((y_pred >= 0).all())
        self.assertTrue((y_pred <= 1).all())
        self.assertTrue(len(np.unique(y_pred)) > 2)


    def test_hgb(self):
        hgb = HGBModel(**{"target_type" : "binary"})
        hgb.fit(self.train_df.drop(columns=self.target_col_name), self.train_df[self.target_col_name])
        y_pred = hgb.predict(self.test_df)
        
        self.assertTrue((y_pred >= 0).all())
        self.assertTrue((y_pred <= 1).all())
        self.assertTrue(len(np.unique(y_pred)) > 2)

    def test_keras_emb(self):
        
        train_csv_path = "/Users/yuwei-1/Documents/projects/Kaggle-tools/data/loan_approval/train.csv"
        test_csv_path = "/Users/yuwei-1/Documents/projects/Kaggle-tools/data/loan_approval/test.csv"
        target_col_name = "loan_status"

        settings = DataSciencePipelineSettings(train_csv_path,
                                               test_csv_path,
                                               target_col_name, 
                                               )
        
        settings = OrdinalEncode.transform(settings)
        settings = ConvertObjectToCategorical.transform(settings)
        
        train_df, test_df = settings.update()
        test_df.drop(columns=[target_col_name], inplace=True)

        
        category_mask = np.array([True if test_df[col].dtype == 'category' else False for col in test_df.columns])
        continuous_feature_idcs = np.where(category_mask == False)[0].tolist()
        categorical_feature_idcs = np.where(category_mask == True)[0].tolist()
        categorical_feature_sizes = test_df.iloc[:, categorical_feature_idcs].nunique().tolist()
        feature_names = test_df.columns.tolist()
        kemb = KerasEmbeddingModel(continuous_feature_idcs=continuous_feature_idcs,
                                   categorical_feature_idcs=categorical_feature_idcs,
                                   categorical_feature_sizes=categorical_feature_sizes,
                                   feature_names=feature_names)
        
        kemb.fit(train_df.drop(columns=target_col_name), train_df[target_col_name])
        y_pred = kemb.predict(test_df)

        self.assertTrue((y_pred >= 0).all())
        self.assertTrue((y_pred <= 1).all())
        self.assertTrue(len(np.unique(y_pred)) > 2)

    def test_keras_fm(self):
        
        train_csv_path = "/Users/yuwei-1/Documents/projects/Kaggle-tools/data/loan_approval/train.csv"
        test_csv_path = "/Users/yuwei-1/Documents/projects/Kaggle-tools/data/loan_approval/test.csv"
        target_col_name = "loan_status"

        settings = DataSciencePipelineSettings(train_csv_path,
                                               test_csv_path,
                                               target_col_name, 
                                               )
        
        # settings = OrdinalEncode.transform(settings)
        # settings = ConvertObjectToCategorical.transform(settings)
        
        train_df, test_df = settings.update()
        test_df.drop(columns=[target_col_name], inplace=True)

        oe = OrdinalEncoder(encoded_missing_value=-1, handle_unknown="use_encoded_value", unknown_value=-1)
        train_df[settings.training_col_names] = oe.fit_transform(train_df[settings.training_col_names])
        test_df[settings.training_col_names] = oe.transform(test_df[settings.training_col_names])
    
        # train_df = train_df[settings.categorical_col_names + [target_col_name]]
        # test_df = test_df[settings.categorical_col_names]
        categorical_feature_sizes = train_df[settings.training_col_names].nunique().tolist()

        kfm = KerasFM(max_features=categorical_feature_sizes,
                      feature_names=test_df.columns.tolist())
        
        kfm.fit(train_df.drop(columns=target_col_name), train_df[target_col_name])
        y_pred = kfm.predict(test_df)

        self.assertTrue((y_pred >= 0).all())
        self.assertTrue((y_pred <= 1).all())
        self.assertTrue(len(np.unique(y_pred)) > 2)

    def test_tabnet(self):
        
        train_csv_path = "/Users/yuwei-1/Documents/projects/Kaggle-tools/data/loan_approval/train.csv"
        test_csv_path = "/Users/yuwei-1/Documents/projects/Kaggle-tools/data/loan_approval/test.csv"
        target_col_name = "loan_status"

        settings = DataSciencePipelineSettings(train_csv_path,
                                               test_csv_path,
                                               target_col_name, 
                                               )
        
        settings = OrdinalEncode.transform(settings)
        settings = ConvertObjectToCategorical.transform(settings)
        
        train_df, test_df = settings.update()
        test_df.drop(columns=[target_col_name], inplace=True)

        
        category_mask = np.array([True if test_df[col].dtype == 'category' else False for col in test_df.columns])
        continuous_feature_idcs = np.where(category_mask == False)[0].tolist()
        categorical_feature_idcs = np.where(category_mask == True)[0].tolist()
        categorical_feature_sizes = test_df.iloc[:, categorical_feature_idcs].nunique().tolist()
        feature_names = test_df.columns.tolist()

        tabnet = TabNetModel(categorical_feature_idcs, categorical_feature_sizes, task="binary")
        tabnet.fit(train_df.drop(columns=target_col_name), train_df[[target_col_name]])
        y_pred = tabnet.predict(test_df)

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