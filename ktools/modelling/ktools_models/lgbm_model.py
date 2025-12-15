from typing import *
import numpy as np
import lightgbm as lgb
from ktools.modelling.base_classes.base_ktools_model import BaseKtoolsModel
from ktools.modelling.base_classes.joblib_saver_mixin import JoblibSaverMixin


class LGBMModel(BaseKtoolsModel, JoblibSaverMixin):
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
        super().__init__(random_state, early_stopping_rounds)
        self._num_boost_round = num_boost_round
        self._verbose = verbose
        self._n_jobs = n_jobs
        self._callbacks = callbacks

        self._lgb_param_grid = {
            "verbose": verbose,
            "random_state": random_state,
            "n_jobs": n_jobs,
            **lgb_param_grid,
        }

        self.is_fitted_ = False

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        validation_set: Union[Tuple[np.ndarray, np.ndarray], None] = None,
        val_size: float = 0.05,
        weights: Union[np.ndarray, None] = None,
    ) -> "LGBMModel":
        X_train, X_valid, y_train, y_valid, weights = self.create_validation(
            X, y, validation_set, weights, val_size
        )

        train_data = lgb.Dataset(X_train, label=y_train, weight=weights)
        eval_sets = [train_data]
        eval_names = ["train"]
        if self._early_stop:
            val_data = lgb.Dataset(X_valid, label=y_valid, reference=train_data)
            eval_sets += [val_data]
            eval_names += ["valid"]
            self._lgb_param_grid["early_stopping_rounds"] = self._early_stopping_rounds

        train_params = {
            "params": self._lgb_param_grid,
            "train_set": train_data,
            "num_boost_round": self._num_boost_round,
            "valid_sets": eval_sets,
            "valid_names": eval_names,
            "callbacks": self._callbacks,
        }

        self.model = lgb.train(**train_params)
        self.is_fitted_ = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        y_pred = self.model.predict(X)
        return y_pred

    @property
    def num_fitted_models(self):
        return self.model.num_trees()
