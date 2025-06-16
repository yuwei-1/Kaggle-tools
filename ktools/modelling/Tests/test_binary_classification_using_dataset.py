from functools import reduce
import unittest
from sklearn.metrics import accuracy_score
from parameterized import parameterized
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from ktools.modelling.Interfaces.i_ktools_model import IKtoolsModel
from ktools.modelling.ktools_models.catboost_model import CatBoostModel
from ktools.modelling.ktools_models.lgbm_model import LGBMModel
from ktools.preprocessing.basic_feature_transformers import *
from ktools.utils.data_science_pipeline_settings import DataSciencePipelineSettings
from ktools.modelling.ktools_models.xgb_model import XGBoostModel


class TestKtoolsBinaryClassificationUsingDataset(unittest.TestCase):
    
    def setUp(self) -> None:
        train_csv_path = "/Users/yuwei-1/Documents/projects/Kaggle-tools/data/loan_approval/train.csv"
        test_csv_path = "/Users/yuwei-1/Documents/projects/Kaggle-tools/data/loan_approval/test.csv"
        self.target_col_name = "loan_status"

        settings = DataSciencePipelineSettings(
            train_csv_path,
            test_csv_path,
            self.target_col_name,
            train_data = pd.read_csv(train_csv_path).sample(1000, random_state=42).reset_index(drop=True)
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
        (XGBoostModel(**{"objective" : "binary:logistic", "early_stopping_rounds" : None}), XGBClassifier(enable_categorical=True)),
        (CatBoostModel(**{"loss_function" : "Logloss",
                          "random_state" : None, 
                          "early_stopping_rounds" : None, 
                          "num_boost_round" : 1000}), CatBoostClassifier()),
        (LGBMModel(**{'objective': 'binary', "early_stopping_rounds" : None}), LGBMClassifier())
    ])
    def test_compare_regression_scores(self, model_cls : IKtoolsModel, original_model):
        if isinstance(original_model, CatBoostClassifier):
            original_model = CatBoostClassifier(cat_features=self.settings.categorical_col_names,
                                               allow_writing_files = False)

        X_train, X_valid, y_train, y_valid = self.data_sets
        model_cls = model_cls.fit(X_train, y_train)
        original_model = original_model.fit(X_train, y_train)

        y_pred_custom = model_cls.predict(X_valid)
        y_pred_expected = original_model.predict(X_valid)

        custom_score = accuracy_score(y_valid, y_pred_custom > 0.5)
        expected_score = accuracy_score(y_valid, y_pred_expected > 0.5)

        self.assertEqual(custom_score, expected_score)