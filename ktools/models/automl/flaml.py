from enum import Enum
from flaml import AutoML
from typing import List, Optional, Tuple, Union
from ktools import logger
from ktools.base.model import BaseKtoolsModel
from ktools.preprocessing.basic_feature_transformers import *
from ktools.utils.helpers import infer_task


T = Union[np.ndarray, pd.DataFrame]


pd_to_np = lambda x: x.to_numpy() if isinstance(x, pd.DataFrame) else x
task_to_default_metric = {
    "regression": "mse",
    "binary_classification": "accuracy",
    "multiclass_classification": "accuracy",
}


class DefaultObjective(Enum):
    regression = "regression"
    binary_classification = "classification"
    multiclass_classification = "classification"


class FLAMLModel(BaseKtoolsModel):
    def __init__(
        self,
        time_budget: float = 5,
        metric: Optional[str] = None,
        task: Optional[str] = None,
        n_jobs: int = -1,
        estimator_list: List[str] = [
            "lgbm",
            "xgboost",
            "xgb_limitdepth",
            "catboost",
            "rf",
            "extra_tree",
        ],
        verbose: int = 0,
        seed: int = 42,
        **extra_params,
    ) -> None:
        super().__init__()
        self._random_state = seed
        self._task = task
        self._time_budget = time_budget
        self._model_parameters = {
            "time_budget": time_budget,
            "metric": metric,
            "task": task,
            "n_jobs": n_jobs,
            "estimator_list": estimator_list,
            "verbose": verbose,
            "seed": seed,
            **extra_params,
        }

    def fit(
        self,
        X: T,
        y: T,
        validation_set: Optional[Tuple[T, T]] = None,
        weights: Optional[T] = None,
    ) -> "FLAMLModel":
        if self._task is None:
            task_id = infer_task(y)
            self._task = task_id
            self._model_parameters["task"] = DefaultObjective[task_id].value
            self._model_parameters["metric"] = task_to_default_metric[task_id]

        if weights is not None:
            logger.warning(
                "Ktools FLAML does not currently support sample weights. Ignoring weights."
            )

        self.model = AutoML(**self._model_parameters)

        fitting_kwargs = {}

        if validation_set is not None:
            X_val, y_val = validation_set
            X_val, y_val = pd_to_np(X_val), pd_to_np(y_val)
            fitting_kwargs["X_val"] = X_val
            fitting_kwargs["y_val"] = y_val

        X, y = pd_to_np(X), pd_to_np(y)
        fitting_kwargs["X_train"] = X
        fitting_kwargs["y_train"] = y

        self.model.fit(
            **fitting_kwargs,
            time_budget=self._time_budget,
        )
        self._fitted = True
        return self

    def predict(self, X: T) -> np.ndarray:
        X = pd_to_np(X)
        if self._task == "regression":
            y_pred = self.model.predict(X)
        elif self._task == "binary_classification":
            y_pred = self.model.predict_proba(X)[:, 1]
        elif self._task == "multiclass_classification":
            y_pred = self.model.predict_proba(X)
        else:
            raise NotImplementedError
        return y_pred
