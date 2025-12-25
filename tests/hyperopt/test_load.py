import pytest
from unittest.mock import MagicMock
from ktools.utils.loader import load_optuna_grid

GRID_CONFIGS = [
    ("ktools/hyperopt/grids/catboost.yml", "base"),
    ("ktools/hyperopt/grids/catboost.yml", "mvs"),
    ("ktools/hyperopt/grids/catboost.yml", "bayesian"),
    ("ktools/hyperopt/grids/catboost.yml", "bernoulli"),
    ("ktools/hyperopt/grids/catboost_early_stopping.yml", "base"),
    ("ktools/hyperopt/grids/catboost_early_stopping.yml", "mvs"),
    ("ktools/hyperopt/grids/catboost_early_stopping.yml", "bayesian"),
    ("ktools/hyperopt/grids/catboost_early_stopping.yml", "bernoulli"),
    ("ktools/hyperopt/grids/lgbm.yml", "base"),
    ("ktools/hyperopt/grids/lgbm.yml", "gbdt-bagging"),
    ("ktools/hyperopt/grids/lgbm.yml", "gbdt-goss"),
    ("ktools/hyperopt/grids/lgbm.yml", "dart"),
    ("ktools/hyperopt/grids/lgbm.yml", "rf"),
    ("ktools/hyperopt/grids/lgbm_early_stopping.yml", "base"),
    ("ktools/hyperopt/grids/lgbm_early_stopping.yml", "gbdt-bagging"),
    ("ktools/hyperopt/grids/lgbm_early_stopping.yml", "gbdt-goss"),
    ("ktools/hyperopt/grids/lgbm_early_stopping.yml", "dart"),
    ("ktools/hyperopt/grids/lgbm_early_stopping.yml", "rf"),
    ("ktools/hyperopt/grids/xgboost.yml", "base"),
    ("ktools/hyperopt/grids/xgboost.yml", "gbtree"),
    ("ktools/hyperopt/grids/xgboost.yml", "gblinear"),
    ("ktools/hyperopt/grids/xgboost.yml", "dart"),
    ("ktools/hyperopt/grids/xgboost_early_stopping.yml", "base"),
    ("ktools/hyperopt/grids/xgboost_early_stopping.yml", "gbtree"),
    ("ktools/hyperopt/grids/xgboost_early_stopping.yml", "gblinear"),
    ("ktools/hyperopt/grids/xgboost_early_stopping.yml", "dart"),
]


@pytest.fixture
def mock_trial():
    """Create a mock Optuna trial that tracks suggested parameters."""
    trial = MagicMock()
    suggested_params = {}

    def suggest_int(name, low, high, **kwargs):
        suggested_params[name] = {"type": "int", "low": low, "high": high, **kwargs}
        return (low + high) // 2

    def suggest_float(name, low, high, **kwargs):
        suggested_params[name] = {"type": "float", "low": low, "high": high, **kwargs}
        return (low + high) / 2

    def suggest_categorical(name, choices, **kwargs):
        suggested_params[name] = {"type": "categorical", "choices": choices, **kwargs}
        return choices[0]

    def set_user_attr(name, value):
        suggested_params[name] = {"type": "fixed", "value": value}
        return value

    trial.suggest_int = suggest_int
    trial.suggest_float = suggest_float
    trial.suggest_categorical = suggest_categorical
    trial.set_user_attr = set_user_attr
    trial.suggested_params = suggested_params

    return trial


@pytest.mark.parametrize("yml_path,model_type", GRID_CONFIGS)
def test_load_grid_and_yaml_anchors_work(yml_path: str, model_type: str, mock_trial):
    """Test that YAML anchors/aliases work and num_boost_round is present."""
    param_grid_getter = load_optuna_grid(yml_path, model_type)

    param_grid_getter(mock_trial)

    assert "num_boost_round" in mock_trial.suggested_params, (
        f"num_boost_round not found in {yml_path} for model_type={model_type}. "
        f"YAML anchor inheritance may have failed. "
        f"Found params: {list(mock_trial.suggested_params.keys())}"
    )

    boost_param = mock_trial.suggested_params["num_boost_round"]
    assert boost_param["type"] in ("int", "categorical"), (
        f"num_boost_round should be int or fixed type, got {boost_param['type']}"
    )


def test_load_grid_and_add_additional_samplers(mock_trial):
    """Test loading grid with extra samplers."""
    extra_samplers = {
        "custom_param": lambda t: t.suggest_float("custom_param", 0.1, 1.0)
    }
    param_grid_getter = load_optuna_grid(
        "ktools/hyperopt/grids/catboost.yml",
        "base",
        extra_samplers=extra_samplers,
    )

    param_grid_getter(mock_trial)

    assert "custom_param" in mock_trial.suggested_params, (
        "custom_param not found in suggested parameters."
    )
    custom_param = mock_trial.suggested_params["custom_param"]
    assert custom_param["type"] == "float", (
        f"custom_param should be float type, got {custom_param['type']}"
    )
