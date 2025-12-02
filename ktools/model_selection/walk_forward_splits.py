import math
import numpy as np
import pandas as pd



class WalkForwardSplit():

    def __init__(self,
                 minimum_training_points : int,
                 testing_points : int,
                 n_splits : int = 5,
                 ) -> None:
        self._minimum_training_points = minimum_training_points
        self._testing_points = testing_points
        self._n_splits = n_splits
    
    def _chunk_lengths(self, L, n):
        sz = math.ceil(L / n)
        return [sz] * (n - 1) + [L - sz * (n - 1)]

    def split(self, X : pd.DataFrame, *args, **kwargs):

        total_size = X.shape[0]
        all_idcs = np.arange(total_size)

        window_region = total_size - (self._minimum_training_points + self._testing_points)
        shift_sizes = [0] + self._chunk_lengths(window_region, self._n_splits - 1)

        train_l, train_r = 0, self._minimum_training_points
        test_l, test_r = self._minimum_training_points, self._minimum_training_points + self._testing_points

        for shift in shift_sizes:

            train_r += shift
            test_l += shift
            test_r += shift

            train_idcs = all_idcs[train_l : train_r]
            test_idcs = all_idcs[test_l : test_r]

            yield (train_idcs, test_idcs)

    def get_n_splits(self):
        return self._n_splits