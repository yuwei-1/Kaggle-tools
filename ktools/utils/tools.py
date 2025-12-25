import numpy as np
from scipy.stats import rankdata


def encode_in_order(array):
    d = {}
    idx = 0
    for i, n in enumerate(array):
        if n not in d:
            d[n] = idx
            idx += 1
        array[i] = d[array[i]]
    return array


def transform_distribution(orig: np.ndarray, target: np.ndarray) -> np.ndarray:
    original_ranks = rankdata(orig, method="ordinal")
    original_quantiles = original_ranks / (original_ranks.shape[0] - 1)
    target_sorted = np.sort(target)
    transformed = np.interp(
        original_quantiles,
        np.linspace(0, 1, target.shape[0], endpoint=True),
        target_sorted,
    )
    return transformed
