import pandas as pd
from ktools.base.preprocessor import BasePreprocessor
from sklearn.preprocessing import StandardScaler
from ktools.config.dataset import DatasetConfig


class StandardScale(BasePreprocessor):
    name = "standard-scaler"

    def __init__(self, config: DatasetConfig) -> None:
        super().__init__(config)
        self.scaler = StandardScaler()

    def fit(self, data: pd.DataFrame) -> "StandardScale":
        self.scaler.fit(data[self.config.numerical_col_names])
        self._fitted = True
        return self

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        copy = data.copy()
        copy[self.config.numerical_col_names] = self.scaler.transform(
            copy[self.config.numerical_col_names]
        )
        return copy
