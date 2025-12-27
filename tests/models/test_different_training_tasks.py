from typing import Tuple
import numpy as np
import pytest
from sklearn.metrics import roc_auc_score
from ktools.base.model import BaseKtoolsModel
from sklearn.datasets import make_regression, make_classification
from sklearn.model_selection import train_test_split
from ktools.models import LGBMModel, XGBoostModel, CatBoostModel
from ktools.models.automl.flaml import FLAMLModel


NUM_MULTICLASS = 5


@pytest.fixture
def dummy_reg_data() -> Tuple[np.ndarray]:
    X, y = make_regression(n_samples=100, n_features=10, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return X_train, X_val, y_train, y_val


@pytest.fixture
def dummy_binclass_data() -> Tuple[np.ndarray]:
    X, y = make_classification(
        n_samples=100, n_features=10, n_classes=2, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return X_train, X_val, y_train, y_val


@pytest.fixture
def dummy_multiclass_data() -> Tuple[np.ndarray]:
    X, y = make_classification(
        n_samples=100,
        n_features=10,
        n_classes=NUM_MULTICLASS,
        n_informative=NUM_MULTICLASS,
        n_clusters_per_class=1,
        random_state=42,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return X_train, X_val, y_train, y_val


@pytest.mark.parametrize(
    "model_cls",
    [
        pytest.param(LGBMModel, id="lightgbm"),
        pytest.param(XGBoostModel, id="xgboost"),
        pytest.param(CatBoostModel, id="catboost"),
        pytest.param(FLAMLModel, id="flaml"),
    ],
)
def test_regression_model(model_cls, dummy_reg_data):
    X_train, X_val, y_train, y_val = dummy_reg_data
    model: BaseKtoolsModel = model_cls()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    nuniques = np.unique(y_pred).shape[0]

    assert nuniques > 0.8 * y_val.shape[0], (
        "Expected high proportion of unique predictions for regression task"
    )


@pytest.mark.parametrize(
    "model_cls",
    [
        pytest.param(LGBMModel, id="lightgbm"),
        pytest.param(XGBoostModel, id="xgboost"),
        pytest.param(CatBoostModel, id="catboost"),
        pytest.param(FLAMLModel, id="flaml"),
    ],
)
def test_binary_classification_model(model_cls, dummy_binclass_data):
    X_train, X_val, y_train, y_val = dummy_binclass_data
    model: BaseKtoolsModel = model_cls()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    # nuniques = np.unique(y_pred).shape[0]
    score = roc_auc_score(y_val, y_pred)

    assert ((y_pred >= 0) & (y_pred <= 1)).all(), "Expected probabilities as output"

    assert score > 0.8, "Expected AUC score to be greater than 0.8"

    assert y_pred.shape == y_val.shape, (
        "Expected prediction shape to match validation labels shape"
    )


@pytest.mark.parametrize(
    "model_cls",
    [
        pytest.param(LGBMModel, id="lightgbm"),
        pytest.param(XGBoostModel, id="xgboost"),
        pytest.param(CatBoostModel, id="catboost"),
        pytest.param(FLAMLModel, id="flaml"),
    ],
)
def test_multiclass_classification_model(model_cls, dummy_multiclass_data):
    X_train, X_val, y_train, y_val = dummy_multiclass_data
    model: BaseKtoolsModel = model_cls()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    # nuniques = np.unique(y_pred).shape[0]
    score = roc_auc_score(y_val, y_pred, multi_class="ovr", average="macro")

    assert ((y_pred >= 0) & (y_pred <= 1)).all(), "Expected probabilities as output"

    assert score > 0.8, "Expected multiclass AUC score to be greater than 0.8"

    assert y_pred.shape[1] == NUM_MULTICLASS, (
        "Expected number of classes in prediction to match training labels"
    )
