from abc import ABC, abstractmethod
from typing import Tuple
import pandas as pd
from ktools.utils.data_science_pipeline_settings import DataSciencePipelineSettings


class IPreprocessingUtility(ABC):

    def __init__(self,
                 data_science_settings : DataSciencePipelineSettings,
                 return_categorical : bool = False,
                 verbose : bool = False) -> None:
        super().__init__()
        self._dss = data_science_settings
        self._return_categorical = return_categorical
        self._verbose = verbose
    
    @abstractmethod
    def process(self) -> Tuple[pd.DataFrame, pd.DataFrame, DataSciencePipelineSettings]:
        pass