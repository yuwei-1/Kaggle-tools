import unittest
import pandas as pd
import numpy as np
from sklearn.datasets import make_regression
from sklearn.metrics import root_mean_squared_error
from gpytorch.means import ConstantMean
from gpytorch.kernels import ScaleKernel, RBFKernel
from ktools.modelling.bayesian.variational_gp import VariationalGP


class TestVariationalGPModel(unittest.TestCase):
    def test_fit_variation_gp(self):
        # Arrange
        expected_rmse = 164.05371959489023
        X, y = make_regression(
            n_samples=100_000, n_features=10, noise=0.1, random_state=42
        )
        feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        X = pd.DataFrame(X, columns=feature_names)
        y = pd.Series(y, name="target")

        model = VariationalGP(
            mean_module=ConstantMean(),
            covariance_module=ScaleKernel(RBFKernel()),
            random_state=42,
            training_epochs=3,
        )

        # Act
        model.fit(X, y)
        y_pred = model.predict(X)
        loss_history = np.array(model.train_loss)
        loss_change = loss_history[1:] - loss_history[:-1]

        # Assert
        self.assertTrue(y_pred.shape == (100_000,))
        self.assertTrue((loss_change < 0).all())
        self.assertAlmostEqual(
            expected_rmse, root_mean_squared_error(y, y_pred), places=6
        )
