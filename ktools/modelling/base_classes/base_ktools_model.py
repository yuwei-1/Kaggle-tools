import numpy as np
from typing import *
from abc import ABC, abstractmethod
from sklearn.model_selection import train_test_split


class BaseKtoolsModel(ABC):
    def __init__(
        self,
        random_state: int,
        early_stopping_rounds: Union[int, None],
    ) -> None:
        self._random_state = random_state
        self._early_stopping_rounds = early_stopping_rounds
        self._early_stop = early_stopping_rounds is not None

    def create_validation(
        self,
        X_train,
        y_train,
        validation_set: Union[Tuple[np.ndarray, np.ndarray], None] = None,
        weights: Union[np.ndarray, None] = None,
        val_size: float = 0.05,
    ):
        create_new_valid = self._early_stop and validation_set is None
        X_valid, y_valid = (None, None) if validation_set is None else validation_set
        if create_new_valid:
            X_train, X_valid, y_train, y_valid = train_test_split(
                X_train, y_train, test_size=val_size, random_state=self._random_state
            )
        weights = np.ones(y_train.shape[0]) if weights is None else weights
        return X_train, X_valid, y_train, y_valid, weights

    @abstractmethod
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        validation_set: Union[Tuple[np.ndarray, np.ndarray], None] = None,
        val_size: float = 0.05,
        weights: Union[np.ndarray, None] = None,
    ) -> "BaseKtoolsModel":
        pass

    @abstractmethod
    def predict(self, X: np.ndarray):
        pass
