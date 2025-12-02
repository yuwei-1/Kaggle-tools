import unittest
import logging
from sys import stdout
import numpy as np
import pandas as pd
from sklearn.datasets import make_regression
from ktools.feature_engineering.deep_feature_creator import DeepFeatureCreator


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logFormatter = logging.Formatter\
("%(name)-12s %(asctime)s %(levelname)-8s %(filename)s:%(funcName)s %(message)s")
consoleHandler = logging.StreamHandler(stdout)
consoleHandler.setFormatter(logFormatter)
logger.addHandler(consoleHandler)

class TestDeepFeatureCreator(unittest.TestCase):

    def test_create_deep_features(self):
        
        # Arrange
        num_features = 8
        repeats = 8
        levels_of_compression = 3
        expected_final_loss = 4.383719444274902
        X, _ = make_regression(n_samples=1000, n_features=num_features, noise=0.1, random_state=42)

        all_feature_names = [f'feature_{i}' for i in range(num_features*repeats)]
        data = pd.DataFrame(np.repeat(X, repeats, axis=-1), columns=all_feature_names)

        train_data = data.iloc[:500]
        test_data = data.iloc[500:]

        # Act
        creator = DeepFeatureCreator(train_data, all_feature_names, levels_of_compression, logger=logger)
        result, _ = creator.create(test_data)
        loss_history = np.array(creator.train_loss)
        loss_change = loss_history[1:] - loss_history[:-1]

        self.assertTrue(result.shape == (500, 64 + 8))
        self.assertTrue((loss_change < 0).all())
        self.assertAlmostEqual(expected_final_loss, loss_history[-1], places=5)