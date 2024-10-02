import pandas as pd
from typing import List
from abc import ABC, abstractmethod


class ISklearnModel(ABC):
    
    @abstractmethod
    def fit(X, y, validation_set : List[pd.DataFrame] = None):
        pass

    @abstractmethod
    def predict(X):
        pass