from enum import Enum
from functools import partial
from sklearn.metrics import accuracy_score
from ktools.metrics.fast_matthew_correlation_coefficient import fast_matthews_corr_coeff


class SupportedMetrics(Enum):
    accuracy = partial(accuracy_score)
    mcc = partial(fast_matthews_corr_coeff)