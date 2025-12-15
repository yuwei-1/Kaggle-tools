import unittest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ktools.model_selection.walk_forward_splits import WalkForwardSplit


class TestWalkForwardSplits(unittest.TestCase):
    def test_split(self):
        # Arrange
        split_obj = WalkForwardSplit(
            minimum_training_points=100_000, testing_points=100_000, n_splits=3
        )
        data_path = "./data/DRW_crypto_price_prediction/train.parquet"

        # Act
        data = pd.read_parquet(data_path)

        i = 0
        splits_data = []
        plt.figure(figsize=(10, 5))
        for train_idx, test_idx in split_obj.split(data):
            splits_data += [train_idx]
            splits_data += [test_idx]
            plt.scatter(train_idx, [i] * train_idx.shape[0], color="green")
            plt.scatter(test_idx, [i] * test_idx.shape[0], color="red")
            i += 1
        plt.savefig(
            "./ktools/model_selection/tests/test_data/plots/test_walk_forward_split.png"
        )
        plt.close()

        all_data = np.concatenate(splits_data)
        # np.save("./ktools/model_selection/tests/test_data/expected_split_idcs", all_data)

        # Assert
        expected_split_idcs = np.load(
            "./ktools/model_selection/tests/test_data/expected_split_idcs.npy"
        )
        self.assertTrue(np.allclose(all_data, expected_split_idcs))
