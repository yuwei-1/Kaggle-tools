import numpy as np
from typing import Union, Optional, Tuple
from abc import ABC, abstractmethod
import pandas as pd


T = Union[np.ndarray, pd.DataFrame]


class BaseKtoolsModel(ABC):
    def __init__(self) -> None:
        self._fitted = False
        self.model = None

    # def create_validation(
    #     self,
    #     X_train,
    #     y_train,
    #     validation_set: Union[Tuple[np.ndarray, np.ndarray], None] = None,
    #     weights: Union[np.ndarray, None] = None,
    #     val_size: float = 0.05,
    # ):
    #     create_new_valid = self._early_stop and validation_set is None
    #     X_valid, y_valid = (None, None) if validation_set is None else validation_set
    #     if create_new_valid:
    #         X_train, X_valid, y_train, y_valid = train_test_split(
    #             X_train, y_train, test_size=val_size, random_state=self._random_state
    #         )
    #     weights = np.ones(y_train.shape[0]) if weights is None else weights
    #     return X_train, X_valid, y_train, y_valid, weights

    @abstractmethod
    def fit(
        self,
        X: T,
        y: T,
        validation_set: Optional[Tuple[T, T]] = None,
        weights: Optional[T] = None,
        val_weights: Optional[T] = None,
    ) -> "BaseKtoolsModel":
        pass

    @abstractmethod
    def predict(self, X: T) -> np.ndarray:
        pass

    @property
    def fitted(self) -> bool:
        return self._fitted
