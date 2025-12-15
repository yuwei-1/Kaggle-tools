import numpy as np


def fast_matthews_corr_coeff(y_true: np.ndarray, y_pred: np.ndarray):
    tp = np.float64(np.sum(y_true & y_pred))
    fn = np.float64(np.sum(y_true & np.logical_not(y_pred)))
    fp = np.float64(np.sum(np.logical_not(y_true) & y_pred))
    tn = np.float64(np.sum(np.logical_not(y_true) & np.logical_not(y_pred)))
    denom = (tp + fn) * (fp + tn) * (tp + fp) * (fn + tn)
    if denom == 0:
        return 0
    else:
        return (tp * tn - fn * fp) / np.sqrt(denom)
