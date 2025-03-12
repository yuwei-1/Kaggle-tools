from typing import List
import pandas as pd
import lightgbm as lgb
import sys
from lightgbm import early_stopping, log_evaluation
from sklearn.model_selection import train_test_split
from typing import List
import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost as xgb
from typing import List
import numpy as np
import pandas as pd
import catboost as cat
from sklearn.model_selection import train_test_split
from catboost import CatBoostRegressor, Pool, train




class LGBMModel():

    def __init__(self,
                 num_boost_round=100,
                 early_stopping_rounds=20,
                 random_state=129,
                 verbose=-1,
                 n_jobs=1,
                 **lgb_param_grid,) -> None:
        super().__init__()
        self._num_boost_round = num_boost_round
        self._lgb_param_grid = {"verbose" : verbose, 
                                "early_stopping_rounds" : early_stopping_rounds,
                                "random_state" : random_state,
                                "n_jobs" : n_jobs,
                                **lgb_param_grid}
        self._callbacks = [
                            # log_evaluation(period=log_period), 
                            # early_stopping(stopping_rounds=stopping_rounds)
                           ]
        self._random_state = random_state
        
    def fit(self, X, y, validation_set = None, val_size=0.05, weights=None):
        if validation_set is None:
            X_train, X_valid, y_train, y_valid = train_test_split(X, 
                                                                  y, 
                                                                  test_size=val_size, 
                                                                  random_state=self._random_state)
        else:
            X_train, y_train = X, y
            X_valid, y_valid = validation_set

        weights = np.ones(y_train.shape[0]) if weights is None else weights
        train_data = lgb.Dataset(X_train, label=y_train, weight=weights)
        val_data = lgb.Dataset(X_valid, label=y_valid, reference=train_data)
        self.model = lgb.train(self._lgb_param_grid,
                                train_data,
                                num_boost_round=self._num_boost_round,
                                valid_sets=[train_data, val_data],
                                valid_names=['train', 'valid'],
                                callbacks=self._callbacks,
                                )
        return self

    def predict(self, X):
        y_pred = self.model.predict(X)
        return y_pred
    


class XGBoostModel():

    def __init__(self,
                 eval_verbosity=False,
                 num_boost_round=100,
                 early_stopping_rounds=20,
                 random_state=129,
                 verbosity=0,
                 n_jobs=1,
                 **xgb_param_grid) -> None:
        super().__init__()
        self._eval_verbosity = eval_verbosity
        self._num_boost_round = num_boost_round
        self._early_stopping_rounds = early_stopping_rounds
        self._xgb_param_grid = {"verbosity" : verbosity,
                                "random_state" : random_state,
                                "n_jobs" : n_jobs,
                                **xgb_param_grid}
        self._random_state = random_state
    
    def fit(self, X, y, validation_set = None, val_size=0.05, weights=None):
        if validation_set is None:
            X_train, X_valid, y_train, y_valid = train_test_split(X, 
                                                                  y, 
                                                                  test_size=val_size, 
                                                                  random_state=self._random_state)
        else:
            X_train, y_train = X, y
            X_valid, y_valid = validation_set

        weights = np.ones(y_train.shape[0]) if weights is None else weights
        train_data = xgb.DMatrix(X_train, label=y_train, enable_categorical=True, weight=weights)
        valid_data = xgb.DMatrix(X_valid, label=y_valid, enable_categorical=True)
        eval_data = [(train_data, 'train'), (valid_data, 'eval')]
    
        self.model = xgb.train(
            self._xgb_param_grid, 
            train_data, 
            evals=eval_data,                       
            early_stopping_rounds=self._early_stopping_rounds,   
            num_boost_round=self._num_boost_round,        
            verbose_eval=self._eval_verbosity                 
        )
        return self

    def predict(self, X):
        test_data = xgb.DMatrix(X, enable_categorical=True)
        y_pred = self.model.predict(test_data)
        return y_pred
    


class CatBoostModel():

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

    def fit(self, X, y, validation_set = None, val_size=0.05, weights=None):
        self.cat_col_names = [col_name for col_name in X.columns if X[col_name].dtype == 'category']

        if validation_set is None:
            X_train, X_valid, y_train, y_valid = train_test_split(X, 
                                                                  y, 
                                                                  test_size=val_size, 
                                                                  random_state=self._random_state)
        else:
            X_train, y_train = X, y
            X_valid, y_valid = validation_set
        
        weights = np.ones(y_train.shape[0]) if weights is None else weights
        train_pool = Pool(data=X_train, label=y_train, cat_features=self.cat_col_names, weight=weights)
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