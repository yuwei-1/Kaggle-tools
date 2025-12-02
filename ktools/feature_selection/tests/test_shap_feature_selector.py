import unittest
import numpy as np
import pandas as pd
from sklearn.datasets import make_regression
from ktools.feature_selection.shap_feature_selector import SHAPFeatureSelector
from ktools.modelling.ktools_models.xgb_model import XGBoostModel


class TestSHAPFeatureSelector(unittest.TestCase):

    def test_select(self):

        # Arrange
        num_informative_features = 5
        X, y, importance = make_regression(n_samples=1000, 
                                           n_features=10, 
                                           noise=0.1, 
                                           n_informative=num_informative_features,
                                           random_state=42,
                                           coef=True)
       
        expected_set_of_features = set(np.argsort(importance).astype(str)[-num_informative_features:])
        tree_model = XGBoostModel()
        tree_model.fit(X, y)

        train_data = pd.DataFrame(X, columns=np.arange(X.shape[1]).astype(str))
        selector = SHAPFeatureSelector(tree_model.model,
                                       top_k_features=10,
                                       feature_importance_save_path="./ktools/feature_selection/tests/test_data/test_shap_feature_importances.png")
        

        # Act
        selected_features, _ = selector.select(train_data)

        # Assert
        top_features_using_shap = set(selected_features[:num_informative_features])
        self.assertEqual(expected_set_of_features, top_features_using_shap)