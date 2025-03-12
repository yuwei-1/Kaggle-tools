from typing import List
import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost as xgb
from ktools.modelling.Interfaces.i_ktools_model import IKtoolsModel
from xgboost import XGBClassifier, XGBRegressor


class XGBoostSKLModel(IKtoolsModel):

    def __init__(self,
                 task = "regression",
                 num_boost_round=100,
                 early_stopping_rounds=20,
                 random_state=129,
                 verbosity=0,
                 n_jobs=1,
                 **xgb_param_grid) -> None:
        
        super().__init__()
        self._num_boost_round = num_boost_round
        self._early_stopping_rounds = early_stopping_rounds
        self._xgb_param_grid = {"verbosity" : verbosity,
                                "random_state" : random_state,
                                "n_jobs" : n_jobs,
                                **xgb_param_grid}
        
        self._verbosity = verbosity
        self._random_state = random_state
        if task == "regression":
            self.model = XGBRegressor(**self._xgb_param_grid, n_estimators=self._num_boost_round, enable_categorical=True)
        else:
            self.model = XGBClassifier(**self._xgb_param_grid, n_estimators=self._num_boost_round, enable_categorical=True)


    def fit(self, X, y, validation_set = None, val_size=0.05):
        if validation_set is None:
            X_train, X_valid, y_train, y_valid = train_test_split(X, 
                                                                  y, 
                                                                  test_size=val_size, 
                                                                  random_state=self._random_state)
        else:
            X_train, y_train = X, y
            X_valid, y_valid = validation_set
        
        eval_data = [(X_train, y_train), (X_valid, y_valid)]

        self.model.fit(X_train, 
                       y_train, 
                       eval_set=eval_data,
                       early_stopping_rounds=self._early_stopping_rounds,
                       verbose=self._verbosity)
    
        return self

    def predict(self, X):
        y_pred = self.model.predict(X)
        return y_pred