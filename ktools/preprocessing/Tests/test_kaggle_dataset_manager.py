import unittest
import pandas as pd
from io import StringIO
from ktools.preprocessing.kaggle_dataset_manager import KaggleDatasetManager



class TestKaggleDatasetManager(unittest.TestCase):

    def setUp(self):

        dataframe = """A,B,C
                       1,2,3
                       4,5,6
                       7,8,9"""
        dataframe = StringIO(dataframe)
        df = pd.read_csv(dataframe)

        training_features = ["A", "B"]
        target = "C"
        training_size = 1/3
        test_size = 1/3
        validation_size = 1/3

        self.dataset_manager = KaggleDatasetManager(df,
                                                    training_features,
                                                    target,
                                                    training_size,
                                                    test_size,
                                                    validation_size)


    def test_partition_dataset(self):
        (X_train, 
        X_valid, 
        X_test, 
        y_train, 
        y_valid, 
        y_test) = self.dataset_manager.dataset_partition()

        self.assertTrue(X_train.shape == (1, 2))
        self.assertTrue(X_valid.shape == (1, 2))
        self.assertTrue(X_test.shape == (1, 2))
        self.assertTrue(y_train.shape == (1,))
        self.assertTrue(y_test.shape == (1,))
        self.assertTrue(y_valid.shape == (1,))

    def test_empty_valid(self):
        self.dataset_manager._validation_size = 0
        self.dataset_manager._training_size = 2/3

        (X_train, 
        X_valid, 
        X_test, 
        y_train, 
        y_valid, 
        y_test) = self.dataset_manager.dataset_partition()

        self.assertTrue(X_train.shape == (2, 2))
        self.assertTrue(X_valid is None)
        self.assertTrue(X_test.shape == (1, 2))
        self.assertTrue(y_train.shape == (2,))
        self.assertTrue(y_test.shape == (1,))
        self.assertTrue(y_valid is None)