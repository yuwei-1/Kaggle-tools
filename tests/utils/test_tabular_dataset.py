from typing import Tuple
import pytest
import numpy as np
import pandas as pd
import torch
from ktools.utils.dataset import TabularDataset


@pytest.fixture
def dummy_data():
    X_num = np.zeros((100, 5))
    X_cat = np.ones((100, 3))
    y = np.zeros(100)
    return X_num, X_cat, y


def test_tabular_dataset_get(dummy_data: Tuple[np.ndarray, np.ndarray, np.ndarray]):
    X_num, X_cat, y = dummy_data
    dataset = TabularDataset(X_num, X_cat, y)

    x_num, x_cat, target = dataset[0]

    assert x_num.shape[0] == X_num.shape[1]
    assert x_cat.shape[0] == X_cat.shape[1]
    assert target.shape[0] == 1
    assert (x_num == 0.0).all()
    assert (x_cat == 1).all()
    assert target == 0.0
    assert x_num.dtype == torch.float32
    assert x_cat.dtype == torch.long


def test_tabular_dataset_len(dummy_data: Tuple[np.ndarray, np.ndarray, np.ndarray]):
    X_num, X_cat, y = dummy_data
    dataset = TabularDataset(X_num, X_cat, y)

    assert len(dataset) == 100


def test_tabular_dataset_without_target(
    dummy_data: Tuple[np.ndarray, np.ndarray, np.ndarray],
):
    X_num, X_cat, _ = dummy_data
    dataset = TabularDataset(X_num, X_cat, y=None)

    result = dataset[0]

    assert len(result) == 2
    x_num, x_cat = result
    assert x_num.shape[0] == X_num.shape[1]
    assert x_cat.shape[0] == X_cat.shape[1]


def test_tabular_dataset_with_dataframes():
    X_num = pd.DataFrame(np.zeros((50, 4)), columns=["a", "b", "c", "d"])
    X_cat = pd.DataFrame(np.ones((50, 2)), columns=["e", "f"])
    y = pd.Series(np.arange(50))

    dataset = TabularDataset(X_num, X_cat, y)

    x_num, x_cat, _ = dataset[0]
    assert x_num.shape[0] == 4
    assert x_cat.shape[0] == 2
    assert x_num.dtype == torch.float32
    assert x_cat.dtype == torch.long
