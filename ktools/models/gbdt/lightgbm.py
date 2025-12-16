from enum import Enum
from typing import *
import numpy as np
import lightgbm as lgb
import pandas as pd
from ktools.base import BaseKtoolsModel, JoblibSaveMixin
from ktools.utils.helpers import infer_task


T = Union[np.ndarray, pd.DataFrame]


class DefaultObjective(Enum):
    regression = "regression"
    binary_classification = "binary"
    multiclass_classification = "multiclass"


class LGBMModel(BaseKtoolsModel, JoblibSaveMixin):
    def __init__(
        self,
        num_boost_round: int = 100,
        early_stopping_rounds: Union[int, None] = 20,
        random_state: int = 129,
        verbose: int = -1,
        n_jobs: int = 1,
        callbacks: List[Any] = [],
        **lgb_param_grid,
    ) -> None:
        super().__init__()
        self._num_boost_round = num_boost_round
        self._verbose = verbose
        self._n_jobs = n_jobs
        self._callbacks = callbacks
        self.early_stopping_rounds = early_stopping_rounds

        self._lgb_param_grid = {
            "verbose": verbose,
            "random_state": random_state,
            "n_jobs": n_jobs,
            **lgb_param_grid,
        }

    def fit(
        self,
        X: T,
        y: T,
        validation_set: Optional[Tuple[T, T]] = None,
        weights: Optional[T] = None,
    ) -> "LGBMModel":
        if "objective" not in self._lgb_param_grid:
            task_id = infer_task(y)
            self._lgb_param_grid["objective"] = DefaultObjective[task_id].value
            if task_id == "multiclass_classification":
                self._lgb_param_grid["num_class"] = np.unique(y).shape[0]

        train_data = lgb.Dataset(X, label=y, weight=weights)
        eval_sets = [train_data]
        eval_names = ["train"]
        if validation_set is not None:
            X_val, y_val = validation_set
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            eval_sets += [val_data]
            eval_names += ["valid"]
            self._lgb_param_grid["early_stopping_rounds"] = self.early_stopping_rounds

        train_params = {
            "params": self._lgb_param_grid,
            "train_set": train_data,
            "num_boost_round": self._num_boost_round,
            "valid_sets": eval_sets,
            "valid_names": eval_names,
            "callbacks": self._callbacks,
        }

        self.model = lgb.train(**train_params)
        self._fitted = True
        return self

    def predict(self, X: T) -> np.ndarray:
        y_pred = self.model.predict(X)
        return y_pred
