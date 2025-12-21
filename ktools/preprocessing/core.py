import pandas as pd
from ktools.base.preprocessor import BasePreprocessor
from ktools.config.dataset import DatasetConfig
from ktools.utils.reduce_dataframe_usage import reduce_dataframe_size


class ReduceMemory(BasePreprocessor):
    def __init__(self, config: DatasetConfig) -> None:
        super().__init__(config)

    def fit(self, data: pd.DataFrame) -> "ReduceMemory":
        return self

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        return reduce_dataframe_size(data)
