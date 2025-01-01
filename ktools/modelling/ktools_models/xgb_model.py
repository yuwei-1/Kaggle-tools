from typing import List
import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost as xgb
from ktools.modelling.Interfaces.i_sklearn_model import ISklearnModel


class XGBoostModel(ISklearnModel):

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
    
    def fit(self, X, y, validation_set = None, val_size=0.05):
        if validation_set is None:
            X_train, X_valid, y_train, y_valid = train_test_split(X, 
                                                                  y, 
                                                                  test_size=val_size, 
                                                                  random_state=self._random_state)
        else:
            X_train, y_train = X, y
            X_valid, y_valid = validation_set
        train_data = xgb.DMatrix(X_train, label=y_train, enable_categorical=True)
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