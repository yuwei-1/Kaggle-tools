from abc import ABC, abstractmethod
from typing import Tuple
import pandas as pd
from ktools.utils.data_science_pipeline_settings import DataSciencePipelineSettings


class IFeatureTransformer(ABC):

    @abstractmethod
    def transform(settings : DataSciencePipelineSettings):
        pass