from typing import List
import numpy as np
import pandas as pd
import catboost as cat
from sklearn.model_selection import train_test_split
from catboost import CatBoostRegressor, Pool, train
from ktools.modelling.Interfaces.i_sklearn_model import ISklearnModel


class CatBoostModel(ISklearnModel):

    def __init__(self,
                 num_boost_round=100,
                 early_stopping_rounds=20,
                 random_state=129,
                 predict_type='prob',
                 verbose=False,
                 **catboost_params) -> None:
        super().__init__()
        self._num_boost_round = num_boost_round
        self._stopping_rounds = early_stopping_rounds
        self._catboost_params = {"random_seed" : random_state,
                                 "verbose" : verbose,
                                 **catboost_params}
        self._random_state = random_state
        self._predict_type = predict_type

    def fit(self, X, y, validation_set = None, val_size=0.05):
        self.cat_col_names = [col_name for col_name in X.columns if X[col_name].dtype == 'category']

        if validation_set is None:
            X_train, X_valid, y_train, y_valid = train_test_split(X, 
                                                                  y, 
                                                                  test_size=val_size, 
                                                                  random_state=self._random_state)
        else:
            X_train, y_train = X, y
            X_valid, y_valid = validation_set
            
        train_pool = Pool(data=X_train, label=y_train, cat_features=self.cat_col_names)
        val_pool = Pool(data=X_valid, label=y_valid, cat_features=self.cat_col_names)
        self.model = cat.train(
                params=self._catboost_params,           
                dtrain=train_pool,   
                eval_set=val_pool,
                num_boost_round=self._num_boost_round,   
                early_stopping_rounds=self._stopping_rounds  
                )
        return self

    def predict(self, X):
        test_pool = Pool(data=X, cat_features=self.cat_col_names)
        if self._predict_type == "prob":
            y_pred = self.model.predict(test_pool, prediction_type='Probability')[:, 1]
        elif self._predict_type == "class":
            y_pred = self.model.predict(test_pool, prediction_type='Class')
        else:
            y_pred = self.model.predict(test_pool)
        return y_pred