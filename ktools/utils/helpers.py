import numpy as np
import pandas as pd
from ktools import logger


def infer_task(y: np.ndarray | pd.Series) -> str:
    """
    Will infer binary, multiclass classification or regression based on the target values.

    Returns:
        0: regression
        1: binary classification
        2: multiclass classification
    """
    if isinstance(y, pd.Series):
        y = y.to_numpy()
    y = y.flatten()

    nuniques = np.unique(y).shape[0]
    has_floats = np.any(y % 1 != 0)

    if has_floats:
        logger.warning("Target contains float values. Inferring regression task.")
        return "regression"
    elif nuniques == 2:
        logger.warning(
            "Target contains two unique values. Inferring binary classification task."
        )
        return "binary_classification"
    elif nuniques > 2:
        logger.warning(
            "Target contains more than two unique values. Inferring multiclass classification task."
        )
        return "multiclass_classification"

    raise ValueError(
        "Unable to infer task type from target values. Is there only one target value?"
    )
