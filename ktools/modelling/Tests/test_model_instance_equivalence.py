import unittest
import numpy as np
import pandas as pd
from ktools.models import XGBoostModel


class TestModelInstanceEquivalence(unittest.TestCase):
    def setUp(self) -> None:
        np.random.seed(0)
        self.X = pd.DataFrame(columns=["A", "B"], data=np.random.randn(100, 2))
        self.y = pd.DataFrame(columns=["C"], data=(self.X["A"] ** 2 + self.X["B"]))

    def test_xgb_model_equivalence(self):
        xgb_model = XGBoostModel()
        xgb_model.fit(self.X, self.y)
        model_1 = xgb_model.model

        xgb_model.fit(self.X, self.y)
        model_2 = xgb_model.model

        self.assertFalse(model_1 == model_2)
