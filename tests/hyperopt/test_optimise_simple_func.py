import numpy as np
from ktools.hyperopt.optuna_optimiser import OptunaHyperparameterOptimizer


def quadratic(x: float):
    return x**2


def test_quadratic_function():
    optimiser = OptunaHyperparameterOptimizer(
        model_type="base",
        grid_yaml_path="tests/hyperopt/grids/test_func.yml",
        n_trials=20,
    )

    optimal_param = optimiser.optimize(tunable_func=quadratic)

    assert isinstance(optimal_param, dict)
    assert "x" in optimal_param
    assert np.isclose(optimal_param["x"], 10.0, atol=0.1)
    assert np.isclose(optimiser.study.best_value, 100.0, atol=1.0)
