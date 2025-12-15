import unittest
import pandas as pd
from sklearn.datasets import make_regression
from ktools.feature_engineering.pca_feature_creator import PCAFeatureCreator


class TestPCAFeatureCreator(unittest.TestCase):
    def test_pca_feature_create(self):
        # Arrange
        num_samples = 1000
        X, y = make_regression(n_samples=num_samples, n_features=10, n_informative=5)
        num_cols = X.shape[1]
        X = pd.DataFrame(
            data=X,
            columns=range(num_cols),
            index=["train"] * int(num_samples / 2) + ["test"] * int(num_samples / 2),
        )
        creator = PCAFeatureCreator(feature_names=list(range(num_cols)))

        # Act
        new_train, _ = creator.reduce(X.loc["train"], X.loc["test"])
        number_pca_features = new_train.shape[1] - num_cols

        # Assert
        self.assertEqual(number_pca_features, 8)
