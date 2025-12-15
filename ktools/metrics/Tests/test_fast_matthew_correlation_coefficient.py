import unittest
import numpy as np
from sklearn.metrics import matthews_corrcoef
from ktools.metrics.fast_matthew_correlation_coefficient import fast_matthews_corr_coeff


class TestFastMatthewCorrCoeff(unittest.TestCase):
    def test_fast_matthew_correlation_coefficient(self):
        rng = np.random.default_rng(42)
        N = 1000000
        err_rate = 0.01
        y_true = np.array([0] * (N // 2) + [1] * (N // 2))
        y_pred = np.where(rng.random(N) < err_rate, 1 - y_true, y_true)

        scikit_implementation = matthews_corrcoef(y_true, y_pred)
        personal_implementation = fast_matthews_corr_coeff(y_true, y_pred)

        print(scikit_implementation, personal_implementation)
        self.assertTrue(scikit_implementation == personal_implementation)
