from typing import List
import pandas as pd
from ktools.base.joblib_mixin import ArtifactSaveMixin
from ktools.base.preprocessor import BasePreprocessor


class PreprocessingPipeline(ArtifactSaveMixin):
    def __init__(self, preprocessors: List[BasePreprocessor]) -> None:
        self.preprocessors = preprocessors

    def train_pipe(self, data: pd.DataFrame) -> pd.DataFrame:
        for preprocessor in self.preprocessors:
            data = preprocessor.fit_transform(data)
        return data

    def inference_pipe(self, data: pd.DataFrame) -> pd.DataFrame:
        for preprocessor in self.preprocessors:
            data = preprocessor.transform(data)
        return data
