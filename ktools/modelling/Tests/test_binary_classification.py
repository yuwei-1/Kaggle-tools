import unittest
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from parameterized import parameterized
from sklearn.model_selection import train_test_split
from ktools.modelling.Interfaces.i_ktools_model import IKtoolsModel
from ktools.modelling.ktools_models.catboost_model import CatBoostModel
from ktools.modelling.ktools_models.lgbm_model import LGBMModel
from ktools.modelling.ktools_models.xgb_model import XGBoostModel


class TestKtoolsBinaryClassification(unittest.TestCase):
    
    def setUp(self) -> None:
        X, y = make_classification(
            n_samples=3000,
            n_features=10,
            n_informative=7,
            n_classes=2,
            random_state=42
        )

        self.X = pd.DataFrame(data=X, columns=np.arange(X.shape[1]))
        self.y = pd.Series(y)

    @parameterized.expand([
        (XGBoostModel(**{"objective" : "binary:logistic", "num_boost_round" : 200, "early_stopping_rounds" : None})),
        (CatBoostModel(**{"loss_function" : "Logloss", "num_boost_round" : 200, "early_stopping_rounds" : None})),
        (LGBMModel(**{'objective': 'binary', "num_boost_round" : 200, "early_stopping_rounds" : None}))
    ])
    def test_binary_classification(self, model_cls : IKtoolsModel):
        X_train, X_valid, y_train, y_valid = train_test_split(self.X, self.y, random_state=42, test_size=0.2)
        model_cls = model_cls.fit(X_train, y_train)
        y_pred = model_cls.predict(X_valid)

        train_score = accuracy_score(y_valid, y_pred > 0.5)
        print(model_cls.num_fitted_models)
        print(train_score)
        self.assertTrue(train_score > 0.9)