from abc import ABC, abstractmethod
import pandas as pd
from ktools.base.joblib_mixin import ArtifactSaveMixin
from ktools.config.dataset import DatasetConfig


class BasePreprocessor(ABC, ArtifactSaveMixin):
    name = "base-preprocessor"

    def __init__(self, config: DatasetConfig):
        self._fitted = False
        self.config = config

    @abstractmethod
    def fit(self, data: pd.DataFrame) -> "BasePreprocessor":
        pass

    @abstractmethod
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        pass

    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        return self.fit(data).transform(data)

    @property
    def fitted(self) -> bool:
        return self._fitted
