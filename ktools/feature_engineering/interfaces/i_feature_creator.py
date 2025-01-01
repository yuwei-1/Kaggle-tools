from typing import List, Tuple
import pandas as pd
from abc import ABC, abstractmethod


class IFeatureCreator(ABC):

    @staticmethod
    @abstractmethod
    def create(df : pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        pass