import numpy as np
from ktools.utils.helpers import infer_task


def test_infer_tricky_binary_classification():
    y = np.array([0.0, 1.0, 0.0, 1.0])
    task_id = infer_task(y)

    assert task_id == "binary_classification", (
        "Expected binary classification task ID (1)"
    )


def test_infer_binary_classification():
    y = np.array([0, 1, 1, 1, 1, 1])
    task_id = infer_task(y)

    assert task_id == "binary_classification", (
        "Expected binary classification task ID (1)"
    )


def test_infer_regression():
    y = np.array([0.0, 1.0, 2.0, 3.0, 4.5, 2, 1, 5.5])
    task_id = infer_task(y)

    assert task_id == "regression", "Expected regression task ID (0)"


def test_infer_tricky_multiclass_classification():
    y = np.array([0.0, 1.0, 2.0, 3.0, 2])
    task_id = infer_task(y)

    assert task_id == "multiclass_classification", (
        "Expected multiclass classification task ID (2)"
    )
