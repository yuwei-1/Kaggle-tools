from collections import defaultdict
from typing import *
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from ktools.modelling.base_classes.base_ktools_model import BaseKtoolsModel
from ktools.modelling.base_classes.joblib_saver_mixin import JoblibSaverMixin


class XGBoostModel(BaseKtoolsModel, JoblibSaverMixin):

    def __init__(self,
                 eval_verbosity: bool = False,
                 num_boost_round: int = 100,
                 early_stopping_rounds: Union[int, None] = 20,
                 random_state: int = 129,
                 verbosity: int = 0,
                 n_jobs: int = 1,
                 **xgb_param_grid) -> None:
        
        super().__init__(random_state, early_stopping_rounds)
        self._eval_verbosity = eval_verbosity
        self._num_boost_round = num_boost_round
        self._verbosity = verbosity
        self._n_jobs = n_jobs

        self._xgb_param_grid = {
            "verbosity": verbosity,
            "random_state": random_state,
            "n_jobs": n_jobs,
            **xgb_param_grid
        }    
        self.is_fitted_ = False
    
    def fit(self, X: np.ndarray, y: np.ndarray, validation_set: Union[Tuple[np.ndarray, np.ndarray], None] = None, 
            val_size: float = 0.05, weights: Union[np.ndarray, None] = None) -> "XGBoostModel":

        X_train, X_valid, y_train, y_valid, weights = self.create_validation(X, y, validation_set, weights, val_size)
        
        train_params = {}
        train_data = xgb.DMatrix(X_train, label=y_train, enable_categorical=True, weight=weights)
        eval_data = [(train_data, 'train')]
        if self._early_stop:
            valid_data = xgb.DMatrix(X_valid, label=y_valid, enable_categorical=True)
            eval_data += [(valid_data, 'eval')]
            train_params['early_stopping_rounds'] = self._early_stopping_rounds

        train_params = {
            "params": self._xgb_param_grid,
            "dtrain": train_data,
            "evals": eval_data,
            "num_boost_round": self._num_boost_round,
            "verbose_eval": self._eval_verbosity,
            **train_params
        }

        self.model = xgb.train(
            **train_params            
        )
        self.is_fitted_ = True
        return self

    def predict(self, X):
        test_data = xgb.DMatrix(X, enable_categorical=True)
        y_pred = self.model.predict(test_data)
        return y_pred