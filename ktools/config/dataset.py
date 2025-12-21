from dataclasses import dataclass
from typing import List, Optional


@dataclass
class DatasetConfig:
    training_col_names: List[str]
    target_col_name: str
    numerical_col_names: List[str]
    categorical_col_names: List[str]
    name: Optional[str] = None
