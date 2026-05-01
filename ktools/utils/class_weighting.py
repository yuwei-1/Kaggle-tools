import numpy as np
import pandas as pd


def _prepare_target(
    y: np.ndarray | pd.Series,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if isinstance(y, pd.Series):
        y = y.to_numpy()

    y = np.asarray(y).flatten()
    if y.size == 0:
        raise ValueError("y must contain at least one target value")

    _, inverse, counts = np.unique(y, return_inverse=True, return_counts=True)
    return y, inverse, counts.astype(float)


def _normalise_sample_weights(
    raw_class_weights: np.ndarray,
    inverse: np.ndarray,
) -> np.ndarray:
    sample_weights = raw_class_weights[inverse]
    return sample_weights / sample_weights.mean()


def class_weight_proportional(y: np.ndarray | pd.Series) -> np.ndarray:
    """
    Return inverse class-frequency weights aligned to the input target.

    Each class weight is inversely proportional to its frequency:
        weight_c = N / (n_classes * count_c)

    Equivalent to sklearn's class_weight='balanced'. The returned vector is
    aligned to ``y`` and normalised so that the mean sample weight equals 1.

    Parameters
    ----------
    y : array-like of shape (n_samples,)
        Integer or string class labels.

    Returns
    -------
    np.ndarray
        Weight vector of shape (n_samples,), aligned to ``y``.
    """
    y, inverse, counts = _prepare_target(y)
    n_classes = counts.size
    raw_class_weights = y.size / (n_classes * counts)
    return _normalise_sample_weights(raw_class_weights, inverse)


def class_weight_log_smoothed(y: np.ndarray | pd.Series) -> np.ndarray:
    """
    Return log-smoothed class weights aligned to the input target.

    Applies a logarithm to dampen extreme weights that arise when class
    counts differ by orders of magnitude:
        weight_c = log(1 + N / count_c)

    The returned vector is aligned to ``y`` and normalised so that the mean
    sample weight equals 1.

    Parameters
    ----------
    y : array-like of shape (n_samples,)
        Integer or string class labels.

    Returns
    -------
    np.ndarray
        Weight vector of shape (n_samples,), aligned to ``y``.
    """
    y, inverse, counts = _prepare_target(y)
    raw_class_weights = np.log1p(y.size / counts)
    return _normalise_sample_weights(raw_class_weights, inverse)


def class_weight_effective_num_samples(
    y: np.ndarray | pd.Series,
    beta: float | None = None,
) -> np.ndarray:
    """
    Return Effective Number of Samples weights aligned to the input target.

    The effective number of samples for class c is:
        E_c = (1 - beta^count_c) / (1 - beta)

    The class weight is the inverse:
        weight_c = 1 / E_c = (1 - beta) / (1 - beta^count_c)

    When ``beta`` is None it is set to (N - 1) / N, the value recommended
    by the original paper.

    The returned vector is aligned to ``y`` and normalised so that the mean
    sample weight equals 1.

    Parameters
    ----------
    y : array-like of shape (n_samples,)
        Integer or string class labels.
    beta : float or None, default None
        Hyperparameter in [0, 1). Controls the degree of smoothing:
        0 gives no re-weighting, values approaching 1 give inverse-frequency
        weighting. When None, defaults to (N - 1) / N.

    Returns
    -------
    np.ndarray
        Weight vector of shape (n_samples,), aligned to ``y``.

    References
    ----------
    Cui, Y., Jia, M., Lin, T.-Y., Song, Y., & Belongie, S. (2019).
    Class-Balanced Loss Based on Effective Number of Samples. CVPR.
    """
    y, inverse, counts = _prepare_target(y)
    n = y.size

    if beta is None:
        beta = (n - 1) / n

    if not (0.0 <= beta < 1.0):
        raise ValueError(f"beta must be in [0, 1), got {beta}")

    effective_num = (1.0 - np.power(beta, counts)) / (1.0 - beta)
    raw_class_weights = 1.0 / effective_num
    return _normalise_sample_weights(raw_class_weights, inverse)
