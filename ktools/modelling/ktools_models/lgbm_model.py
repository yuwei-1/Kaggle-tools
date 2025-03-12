from typing import List
import numpy as np
import pandas as pd
import lightgbm as lgb
import sys

sys.path.append("/Users/yuwei-1/Documents/projects/Kaggle-tools")
from lightgbm import early_stopping, log_evaluation
from ktools.modelling.Interfaces.i_ktools_model import IKtoolsModel
from sklearn.model_selection import train_test_split

from ktools.utils.data_science_pipeline_settings import DataSciencePipelineSettings


class LGBMModel(IKtoolsModel):

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