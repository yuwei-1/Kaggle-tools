import unittest
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from ktools.fitting.cross_validate_then_test_sklearn_model import CrossValidateTestSklearnModel



class TestCrossValidateTestSklearnModel(unittest.TestCase):

    def setUp(self):
        model = LinearRegression()
        np.random.seed(0)
        X = pd.DataFrame(columns=["A", "B"], data=np.random.randn(100, 2))
        y = pd.DataFrame(columns=["C"], data=(X["A"]**2 + X["B"]))
        num_splits = 5
        # y = np.where(y > 0, 1, 0)

        train_idcs = np.arange(80)
        test_idcs = np.arange(80, 100)

        self.X_train, self.y_train = X.iloc[train_idcs], y.iloc[train_idcs]
        self.X_test, self.y_test = X.iloc[test_idcs], y.iloc[test_idcs]

        eval_metrics = {"r2" : r2_score}

        self.eval_tool = CrossValidateTestSklearnModel(model,
                                                       eval_metrics,
                                                       KFold(num_splits),
                                                       num_splits)

    def test_evaluate(self):
        
        model, cv_scores, test_scores = self.eval_tool.evaluate(self.X_train,
                                                                self.y_train,
                                                                self.X_test,
                                                                self.y_test)
        
        self.assertIsInstance(model, LinearRegression)
        self.assertTrue(np.allclose(np.array([-0.077,
                                              0.601,
                                              0.539,
                                              0.422,
                                              0.427]), cv_scores.values.squeeze(), atol=0.001))
        self.assertTrue({'r2': 0.7389520959653514} == test_scores)