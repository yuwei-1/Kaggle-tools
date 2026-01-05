import os
from typing import Any, Optional, Tuple, Union
import dotenv
import numpy as np
import pandas as pd
from tabpfn import TabPFNClassifier, TabPFNRegressor

from ktools.base.model import BaseKtoolsModel
from ktools.base.joblib_mixin import JoblibSaveMixin
from ktools.utils.helpers import infer_task


T = Union[np.ndarray, pd.DataFrame]
dotenv.load_dotenv()


class TabPFNModel(BaseKtoolsModel, JoblibSaveMixin):
    MAX_SAMPLES = 50000

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "auto",
        random_state: int = 129,
        **tabpfn_params: Any,
    ) -> None:
        super().__init__()
        self._model_path = model_path
        self._device = device
        self._random_state = random_state
        self._tabpfn_params = tabpfn_params
        self._task: Optional[str] = None

    def fit(
        self,
        X: T,
        y: T,
        validation_set: Optional[Tuple[T, T]] = None,
        weights: Optional[T] = None,
        val_weights: Optional[T] = None,
    ) -> "TabPFNModel":
        self._task = infer_task(y)

        model_kwargs = {
            "device": self._device,
            "random_state": self._random_state,
            **self._tabpfn_params,
        }

        if self._task == "regression":
            if self._model_path is None:
                model_kwargs["model_path"] = os.getenv("TABPFN_REGRESSOR_PATH")
            self.model = TabPFNRegressor(**model_kwargs)
        else:
            if self._model_path is None:
                model_kwargs["model_path"] = os.getenv("TABPFN_CLASSIFIER_PATH")
            self.model = TabPFNClassifier(**model_kwargs)

        self.model.fit(X, y)
        self._fitted = True
        return self

    def predict(self, X: T) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model is not fitted yet. Please call 'fit' first.")

        inference_size = X.shape[0]

        if self._task == "regression":
            pred_func = lambda x: self.model.predict(x)
        elif self._task == "binary_classification":
            pred_func = lambda x: self.model.predict_proba(x)[:, 1]
        else:
            pred_func = lambda x: self.model.predict_proba(x)

        y_preds = []
        for start in range(0, inference_size, self.MAX_SAMPLES):
            end = min(start + self.MAX_SAMPLES, inference_size)
            X_chunk = X[start:end]
            y_pred_chunk = pred_func(X_chunk)
            y_preds.append(y_pred_chunk)
        y_pred = np.concatenate(y_preds, axis=0)

        return y_pred
