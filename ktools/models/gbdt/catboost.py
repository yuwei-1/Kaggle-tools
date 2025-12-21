from enum import Enum
from typing import Any, Dict, Union, Optional, Tuple
import numpy as np
import catboost as cat
from catboost import Pool
import pandas as pd
from ktools.base import BaseKtoolsModel, JoblibSaveMixin
from ktools.utils.helpers import infer_task


T = Union[np.ndarray, pd.DataFrame]


class DefaultObjective(Enum):
    regression = "RMSE"
    binary_classification = "Logloss"
    multiclass_classification = "MultiClass"


class CatBoostModel(BaseKtoolsModel, JoblibSaveMixin):
    def __init__(
        self,
        num_boost_round: int = 100,
        early_stopping_rounds: Optional[int] = 20,
        random_state: int = 129,
        verbose: bool = False,
        allow_writing_files: bool = False,
        **catboost_params,
    ) -> None:
        super().__init__()
        self.model: Union[cat.CatBoost, None] = None
        self._classifier: bool = False
        self._num_boost_round = num_boost_round
        self._verbose = verbose
        self._allow_writing_files = allow_writing_files
        self._early_stopping_rounds = early_stopping_rounds

        self._catboost_params = {
            "random_seed": random_state,
            "verbose": verbose,
            "allow_writing_files": allow_writing_files,
            **catboost_params,
        }

    def fit(
        self,
        X: T,
        y: T,
        validation_set: Optional[Tuple[T, T]] = None,
        weights: Optional[T] = None,
    ) -> "CatBoostModel":
        task_id = infer_task(y)
        self._classifier = task_id != "regression"
        if "loss_function" not in self._catboost_params:
            self._catboost_params["loss_function"] = DefaultObjective[task_id].value

        self.cat_col_names = (
            [col for col in X.columns if X[col].dtype == "category"]
            if isinstance(X, pd.DataFrame)
            else []
        )
        train_params: Dict[Any, Any] = {"eval_set": None}
        train_pool = Pool(
            data=X, label=y, cat_features=self.cat_col_names, weight=weights
        )
        if validation_set is not None:
            X_val, y_val = validation_set
            train_params["eval_set"] = Pool(
                data=X_val, label=y_val, cat_features=self.cat_col_names
            )
            train_params["early_stopping_rounds"] = self._early_stopping_rounds

        train_params = {
            "params": self._catboost_params,
            "dtrain": train_pool,
            "num_boost_round": self._num_boost_round,
            **train_params,
        }
        self.model = cat.train(**train_params)
        self._fitted = True
        return self

    def predict(self, X: T) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model is not fitted yet. Please call 'fit' first.")
        test_pool = Pool(data=X, cat_features=self.cat_col_names)
        if self._classifier:
            y_pred = self.model.predict(test_pool, prediction_type="Probability")[:, 1]
        else:
            y_pred = self.model.predict(test_pool)
        return y_pred

        # if self._predict_type == "prob":
        #     y_pred = self.model.predict(test_pool, prediction_type="Probability")[:, 1]
        # elif self._predict_type == "class":
        #     y_pred = self.model.predict(test_pool, prediction_type="Class")
        # else:
