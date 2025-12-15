from typing import Callable, Union
import numpy as np
import pandas as pd
from abc import abstractmethod, ABC


class IEnsembleMethod(ABC):
    def __init__(
        self,
        oof_dataframe: pd.DataFrame,
        train_labels: Union[pd.DataFrame, pd.Series, np.ndarray],
        metric: Callable,
    ) -> None:
        self._oofs = oof_dataframe
        self._labels = train_labels
        self._metric = metric

    @abstractmethod
    def fit_weights(self):
        pass

    @abstractmethod
    def predict(self):
        pass
