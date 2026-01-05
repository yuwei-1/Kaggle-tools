from typing import Any, List, Optional, Tuple, Union
import math
import numpy as np
import pandas as pd
from pytorch_tabnet.tab_model import TabNetClassifier, TabNetRegressor

from ktools.base.model import BaseKtoolsModel
from ktools.base.joblib_mixin import JoblibSaveMixin
from ktools.utils.helpers import infer_task


T = Union[np.ndarray, pd.DataFrame]


def default_emb_dim_fn(cardinality: int) -> int:
    """Default embedding dimension function: sqrt of cardinality, min 2, max 50."""
    return max(2, min(50, int(math.sqrt(cardinality))))


class TabNetModel(BaseKtoolsModel, JoblibSaveMixin):
    """
    TabNet wrapper following the BaseKtoolsModel interface.

    TabNet is a deep learning architecture for tabular data that uses
    sequential attention to choose which features to reason from at each
    decision step.

    Categorical features are automatically inferred from the data during fit().
    """

    def __init__(
        self,
        # Fit parameters
        batch_size: int = 32,
        virtual_batch_size: int = 128,
        num_workers: int = 0,
        drop_last: bool = False,
        max_epochs: int = 200,
        patience: int = 50,
        eval_metric: Optional[List[str]] = None,
        # All other TabNet params (n_d, n_a, n_steps, gamma, seed, verbose, etc.)
        **tabnet_params: Any,
    ) -> None:
        super().__init__()

        # These will be inferred during fit()
        self._cat_idxs: List[int] = []
        self._cat_dims: List[int] = []
        self._cat_emb_dim: List[int] = []

        # Fit parameters
        self._batch_size = batch_size
        self._virtual_batch_size = virtual_batch_size
        self._num_workers = num_workers
        self._drop_last = drop_last
        self._max_epochs = max_epochs
        self._patience = patience
        self._eval_metric = eval_metric

        # TabNet model parameters
        self._tabnet_params = tabnet_params
        self._task: Optional[str] = None

    def _get_model_kwargs(self) -> dict:
        """Build kwargs dict for TabNet model initialization."""
        return {
            "cat_idxs": self._cat_idxs,
            "cat_dims": self._cat_dims,
            "cat_emb_dim": self._cat_emb_dim,
            **self._tabnet_params,
        }

    def _convert_to_numpy(self, data: T) -> np.ndarray:
        """Convert DataFrame or array to numpy array."""
        if isinstance(data, pd.DataFrame):
            return data.values
        elif isinstance(data, pd.Series):
            return data.values.reshape(-1, 1)
        elif len(data.shape) == 1:
            return data.reshape(-1, 1)
        return data

    def _infer_categorical_features(self, X: T) -> None:
        """
        Infer categorical feature indices and dimensions from the data.

        Categorical features are identified as:
        - Columns with dtype 'category', 'object', 'bool'
        - Integer columns with low cardinality (< 20 unique values)
        """
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)

        self._cat_idxs = []
        self._cat_dims = []
        self._cat_emb_dim = []

        for idx, col in enumerate(X.columns):
            dtype = X[col].dtype
            n_unique = X[col].nunique()

            is_categorical = (
                dtype.name == "category" or dtype == object or dtype == bool
            )

            if is_categorical:
                self._cat_idxs.append(idx)
                # Add 1 to handle potential unseen categories
                cardinality = n_unique + 1
                self._cat_dims.append(cardinality)
                self._cat_emb_dim.append(default_emb_dim_fn(cardinality))

    def fit(
        self,
        X: T,
        y: T,
        validation_set: Optional[Tuple[T, T]] = None,
        weights: Optional[T] = None,
        val_weights: Optional[T] = None,
    ) -> "TabNetModel":
        """
        Fit the TabNet model.

        Args:
            X: Training features.
            y: Training target.
            validation_set: Optional tuple of (X_val, y_val) for early stopping.
            weights: Optional sample weights for training data.
            val_weights: Optional sample weights for validation data (unused).

        Returns:
            self: The fitted model.
        """
        # Infer categorical features from data
        self._infer_categorical_features(X)

        self._task = infer_task(y)
        model_kwargs = self._get_model_kwargs()

        # Convert data to numpy
        X_train = self._convert_to_numpy(X)
        y_train = self._convert_to_numpy(y)

        # Initialize appropriate model based on task
        if self._task == "regression":
            self.model = TabNetRegressor(**model_kwargs)
            default_metric = ["rmse"]
        else:
            y_train = y_train.squeeze()
            self.model = TabNetClassifier(**model_kwargs)
            default_metric = (
                ["auc"] if self._task == "binary_classification" else ["accuracy"]
            )

        eval_metric = (
            self._eval_metric if self._eval_metric is not None else default_metric
        )

        # Prepare fit kwargs
        fit_kwargs = {
            "X_train": X_train,
            "y_train": y_train,
            "eval_metric": eval_metric,
            "batch_size": self._batch_size,
            "virtual_batch_size": self._virtual_batch_size,
            "num_workers": self._num_workers,
            "drop_last": self._drop_last,
            "max_epochs": self._max_epochs,
            "patience": self._patience,
        }

        # Add validation set if provided
        if validation_set is not None:
            X_val, y_val = validation_set
            X_val = self._convert_to_numpy(X_val)
            y_val = self._convert_to_numpy(y_val)
            fit_kwargs["eval_set"] = [(X_val, y_val)]
            fit_kwargs["eval_name"] = ["val"]

        # Add sample weights if provided
        if weights is not None:
            fit_kwargs["weights"] = 1  # Use sample weights
            # TabNet expects weights as a separate parameter during fit

        self.model.fit(**fit_kwargs)
        self._fitted = True
        return self

    def predict(self, X: T) -> np.ndarray:
        """
        Make predictions using the fitted model.

        Args:
            X: Features to predict on.

        Returns:
            Predictions as numpy array. For classification, returns probabilities.
        """
        if self.model is None:
            raise ValueError("Model is not fitted yet. Please call 'fit' first.")

        X = self._convert_to_numpy(X)

        if self._task == "regression":
            y_pred = self.model.predict(X)
        elif self._task == "binary_classification":
            y_pred = self.model.predict_proba(X)[:, 1]
        else:
            y_pred = self.model.predict_proba(X)

        return y_pred
