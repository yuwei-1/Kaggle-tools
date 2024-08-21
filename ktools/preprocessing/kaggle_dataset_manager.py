import pandas as pd
import numpy as np
from typing import List
from sklearn.model_selection import train_test_split


class KaggleDatasetManager:

    def __init__(self,
                 dataframe : pd.DataFrame,
                 training_features : List[str],
                 target : str,
                 training_size : float = 0.8,
                 test_size : float = 0.2,
                 validation_size : float = 0,
                 random_state : int = 42) -> None:
        
        self.dataframe = dataframe
        self._training_features = training_features
        self._target = target
        self._training_size = training_size
        self._test_size = test_size
        self._validation_size = validation_size
        self._random_state = random_state
        self._dataset_size_guard(training_size, test_size, validation_size)

    def dataset_partition(self):
        X_valid, y_valid = None, None
        X_train, X_test, y_train, y_test = train_test_split(self.dataframe[self._training_features], 
                                                            self.dataframe[self._target], 
                                                            test_size=self._test_size, 
                                                            random_state=self._random_state)
        if self._validation_size:
            proportion_of_valid = self._validation_size/(self._validation_size + self._training_size)
            X_train, X_valid, y_train, y_valid = train_test_split(X_train, 
                                                                  y_train, 
                                                                  test_size=proportion_of_valid, 
                                                                  random_state=self._random_state)
        return X_train, X_valid, X_test, y_train, y_valid, y_test

    @staticmethod
    def _dataset_size_guard(*sizes):
        assert np.allclose(np.sum([*sizes]), 1, atol=0.1), "dataset sizes must add to one"