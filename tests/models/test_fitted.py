import pytest
from ktools.base.model import BaseKtoolsModel
from sklearn.datasets import make_regression
from ktools.models import LGBMModel, XGBoostModel, CatBoostModel
from ktools.models.automl.flaml import FLAMLModel


@pytest.fixture
def dummy_data():
    X, y = make_regression(n_samples=100, n_features=10, random_state=42)
    return X, y


@pytest.mark.parametrize(
    "model_cls",
    [
        pytest.param(LGBMModel, id="lightgbm"),
        pytest.param(XGBoostModel, id="xgboost"),
        pytest.param(CatBoostModel, id="catboost"),
        pytest.param(FLAMLModel, id="flaml"),
    ],
)
def test_model_fitted(model_cls, dummy_data):
    model: BaseKtoolsModel = model_cls()
    assert not model.fitted
    model.fit(*dummy_data)
    assert model.fitted
