from typing import List
import numpy as np
import pandas as pd
import catboost as cat
from sklearn.model_selection import train_test_split
from catboost import CatBoostRegressor, Pool, train
from ktools.modelling.Interfaces.i_ktools_model import IKtoolsModel


class CatBoostModel(IKtoolsModel):

    def __init__(self,
                 num_boost_round=100,
                 early_stopping_rounds=20,
                 random_state=129,
                 predict_type='else',
                 verbose=False,
                 **catboost_params) -> None:
        super().__init__()
        self._num_boost_round = num_boost_round
        self._random_state = random_state
        self._predict_type = predict_type
        self._early_stopping_rounds = {'early_stopping_rounds' : early_stopping_rounds} \
                                        if early_stopping_rounds else {}

        self._catboost_params = {"random_seed" : random_state,
                                 "verbose" : verbose,
                                 **catboost_params}


    def fit(self, X, y, validation_set = None, val_size=0.05, weights=None):
        self.cat_col_names = [col_name for col_name in X.columns if X[col_name].dtype == 'category']

        early_stop = len(self._early_stopping_rounds) != 0
        X_train, y_train = X, y
        if validation_set is not None: X_valid, y_valid = validation_set            
            
        if early_stop and validation_set is None:
            X_train, X_valid, y_train, y_valid = train_test_split(X, 
                                                                  y, 
                                                                  test_size=val_size, 
                                                                  random_state=self._random_state)
        weights = np.ones(y_train.shape[0]) if weights is None else weights
        
        train_pool = Pool(data=X_train, label=y_train, cat_features=self.cat_col_names, weight=weights)
        if early_stop:
            val_pool = Pool(data=X_valid, label=y_valid, cat_features=self.cat_col_names)
        else:
            val_pool = None
        self.model = cat.train(
                params=self._catboost_params,           
                dtrain=train_pool,   
                eval_set=val_pool,
                num_boost_round=self._num_boost_round,   
                **self._early_stopping_rounds  
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