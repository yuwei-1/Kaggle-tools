import pytest
from ktools.utils.loader import load_optuna_grid


@pytest.mark.parametrize(
    "yml_path",
    [
        "ktools/hyperopt/grids/lgbm.yml",
    ],
)
def test_load(yml_path: str):
    param_grid = load_optuna_grid(yml_path, "base")
