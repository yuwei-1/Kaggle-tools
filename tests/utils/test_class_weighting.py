import math

import numpy as np
import pandas as pd
import pytest

from ktools.utils.class_weighting import (
    class_weight_effective_num_samples,
    class_weight_log_smoothed,
    class_weight_proportional,
)


def _assert_weight_vector_matches_input(y, weights):
    assert isinstance(weights, np.ndarray)
    assert weights.shape == np.asarray(y).shape
    assert np.all(weights > 0)
    assert weights.mean() == pytest.approx(1.0)


HAPPY_PATH_CASES = [
    pytest.param(
        class_weight_proportional,
        np.array([0, 0, 1]),
        {},
        np.array([0.75, 0.75, 1.5]),
        id="proportional-imbalanced-binary",
    ),
    pytest.param(
        class_weight_log_smoothed,
        np.array([0, 0, 1]),
        {},
        np.array(
            [
                math.log(2.5),
                math.log(2.5),
                math.log(4.0),
            ]
        )
        / ((2 * math.log(2.5) + math.log(4.0)) / 3),
        id="log-smoothed-imbalanced-binary",
    ),
    pytest.param(
        class_weight_effective_num_samples,
        np.array([0, 0, 1]),
        {"beta": 0.5},
        np.array([6 / 7, 6 / 7, 9 / 7]),
        id="effective-num-imbalanced-binary",
    ),
]


@pytest.mark.parametrize("weight_func,y,kwargs,expected", HAPPY_PATH_CASES)
def test_class_weight_vector_happy_path(weight_func, y, kwargs, expected):
    weights = weight_func(y, **kwargs)

    _assert_weight_vector_matches_input(y, weights)
    np.testing.assert_allclose(weights, expected)


@pytest.mark.parametrize(
    "weight_func,y,kwargs,expected",
    [
        pytest.param(
            class_weight_proportional,
            np.array([1, 1, 1]),
            {},
            np.ones(3),
            id="proportional-single-class",
        ),
        pytest.param(
            class_weight_log_smoothed,
            np.array([1, 1, 1]),
            {},
            np.ones(3),
            id="log-smoothed-single-class",
        ),
        pytest.param(
            class_weight_effective_num_samples,
            np.array([1, 1, 1]),
            {},
            np.ones(3),
            id="effective-num-single-class",
        ),
        pytest.param(
            class_weight_effective_num_samples,
            np.array([0, 0, 1]),
            {"beta": 0.0},
            np.ones(3),
            id="effective-num-beta-zero-no-reweighting",
        ),
    ],
)
def test_class_weight_vector_edge_cases(weight_func, y, kwargs, expected):
    weights = weight_func(y, **kwargs)

    _assert_weight_vector_matches_input(y, weights)
    np.testing.assert_allclose(weights, expected)


@pytest.mark.parametrize(
    "weight_func,kwargs,input_y,expected",
    [
        pytest.param(
            class_weight_proportional,
            {},
            pd.Series(["dog", "cat", "dog", "bird"]),
            np.array([2 / 3, 4 / 3, 2 / 3, 4 / 3]),
            id="proportional-string-label-series",
        ),
        pytest.param(
            class_weight_log_smoothed,
            {},
            pd.Series(["dog", "cat", "dog", "bird"]),
            None,
            id="log-smoothed-string-label-series",
        ),
        pytest.param(
            class_weight_effective_num_samples,
            {"beta": 0.5},
            pd.Series(["dog", "cat", "dog", "bird"]),
            np.array([0.8, 1.2, 0.8, 1.2]),
            id="effective-num-string-label-series",
        ),
    ],
)
def test_class_weight_vector_preserves_input_order(
    weight_func, kwargs, input_y, expected
):
    weights = weight_func(input_y, **kwargs)

    _assert_weight_vector_matches_input(input_y, weights)

    if expected is not None:
        np.testing.assert_allclose(weights, expected)

    assert weights[0] == pytest.approx(weights[2])
    assert weights[1] == pytest.approx(weights[3])
    assert weights[1] > weights[0]


@pytest.mark.parametrize(
    "weight_func,kwargs",
    [
        pytest.param(class_weight_proportional, {}, id="proportional-empty-target"),
        pytest.param(class_weight_log_smoothed, {}, id="log-smoothed-empty-target"),
        pytest.param(
            class_weight_effective_num_samples,
            {},
            id="effective-num-empty-target",
        ),
    ],
)
def test_class_weight_vector_raises_for_empty_target(weight_func, kwargs):
    with pytest.raises(ValueError, match="at least one target value"):
        weight_func(np.array([]), **kwargs)


@pytest.mark.parametrize(
    "beta",
    [
        pytest.param(-0.1, id="negative-beta"),
        pytest.param(1.0, id="beta-equals-one"),
    ],
)
def test_effective_num_samples_raises_for_invalid_beta(beta):
    with pytest.raises(ValueError, match=r"beta must be in \[0, 1\)"):
        class_weight_effective_num_samples(np.array([0, 0, 1]), beta=beta)


def test_log_smoothed_and_effective_num_reduce_extreme_imbalance_ratio():
    y = np.array([0] * 1000 + [1])

    proportional = class_weight_proportional(y)
    log_smoothed = class_weight_log_smoothed(y)
    effective_num = class_weight_effective_num_samples(y)

    proportional_ratio = proportional[-1] / proportional[0]
    log_ratio = log_smoothed[-1] / log_smoothed[0]
    effective_num_ratio = effective_num[-1] / effective_num[0]

    assert log_ratio < proportional_ratio
    assert effective_num_ratio < proportional_ratio
