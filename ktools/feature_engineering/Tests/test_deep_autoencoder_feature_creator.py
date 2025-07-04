import unittest
import numpy as np
import pandas as pd
from sklearn.datasets import make_regression
from ktools.feature_engineering.deep_feature_creator import DeepFeatureCreator



class TestDeepFeatureCreator(unittest.TestCase):

    def test_create_deep_features(self):

        num_features = 8
        repeats = 8
        levels_of_compression = 3
        X, _ = make_regression(n_samples=1000, n_features=num_features, noise=0.1, random_state=42)

        all_feature_names = [f'feature_{i}' for i in range(num_features*repeats)]
        data = pd.DataFrame(np.repeat(X, repeats, axis=-1), columns=all_feature_names)

        train_data = data.iloc[:500]
        test_data = data.iloc[500:]

        creator = DeepFeatureCreator(train_data, all_feature_names, levels_of_compression)
        result = creator.create(test_data)

        self.assertTrue(result.shape == (500, 64 + 8))