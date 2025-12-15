import numpy as np
from typing import *
from abc import ABC, abstractmethod


class IKtoolsModel(ABC):
    @abstractmethod
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        validation_set: Union[Tuple[np.ndarray, np.ndarray], None] = None,
        val_size: float = 0.05,
        weights: Union[np.ndarray, None] = None,
    ) -> "IKtoolsModel":
        pass

    @abstractmethod
    def predict(X):
        pass

    @property
    def num_fitted_models(self):
        return None
