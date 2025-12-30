from enum import Enum
from typing import *
import numpy as np
import pandas as pd
import xgboost as xgb
from ktools.base.model import BaseKtoolsModel
from ktools.base.joblib_mixin import JoblibSaveMixin
from ktools.utils.helpers import infer_task


T = Union[np.ndarray, pd.DataFrame]


class DefaultObjective(Enum):
    regression = "reg:squarederror"
    binary_classification = "binary:logistic"
    multiclass_classification = "multi:softprob"


class XGBoostModel(BaseKtoolsModel, JoblibSaveMixin):
    def __init__(
        self,
        eval_verbosity: bool = False,
        num_boost_round: int = 100,
        early_stopping_rounds: Union[int, None] = 20,
        random_state: int = 129,
        verbosity: int = 0,
        n_jobs: int = 1,
        **xgb_param_grid,
    ) -> None:
        super().__init__()
        self._eval_verbosity = eval_verbosity
        self._num_boost_round = num_boost_round
        self._verbosity = verbosity
        self._n_jobs = n_jobs
        self._early_stopping_rounds = early_stopping_rounds

        self._xgb_param_grid = {
            "verbosity": verbosity,
            "random_state": random_state,
            "n_jobs": n_jobs,
            **xgb_param_grid,
        }

    def fit(
        self,
        X: T,
        y: T,
        validation_set: Optional[Tuple[T, T]] = None,
        weights: Optional[T] = None,
        val_weights: Optional[T] = None,
    ) -> "XGBoostModel":
        train_params = {}
        if "objective" not in self._xgb_param_grid:
            task_id = infer_task(y)
            self._xgb_param_grid["objective"] = DefaultObjective[task_id].value
            if task_id == "multiclass_classification":
                self._xgb_param_grid["num_class"] = np.unique(y).shape[0]

        train_data = xgb.DMatrix(X, label=y, enable_categorical=True, weight=weights)
        eval_data = [(train_data, "train")]
        if validation_set is not None:
            X_val, y_val = validation_set
            valid_data = xgb.DMatrix(
                X_val, label=y_val, enable_categorical=True, weight=val_weights
            )
            eval_data = [(valid_data, "eval")]
            train_params["early_stopping_rounds"] = self._early_stopping_rounds

        train_params = {
            "params": self._xgb_param_grid,
            "dtrain": train_data,
            "evals": eval_data,
            "num_boost_round": self._num_boost_round,
            "verbose_eval": self._eval_verbosity,
            **train_params,
        }

        self.model = xgb.train(**train_params)
        self._fitted = True
        return self

    def predict(self, X: T) -> np.ndarray:
        test_data = xgb.DMatrix(X, enable_categorical=True)
        y_pred = self.model.predict(test_data)
        return y_pred
