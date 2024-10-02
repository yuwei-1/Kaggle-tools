import unittest
import pandas as pd
import numpy as np
import optuna
from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn.linear_model import Lasso
from sklearn.model_selection import KFold
from ktools.fitting.cross_validation_executor import CrossValidationExecutor
from ktools.hyperparameter_optimization.i_model_param_grid import IModelParamGrid
from ktools.hyperparameter_optimization.model_param_grids import BaseCatBoostParamGrid, BaseLGBMParamGrid, BaseXGBoostParamGrid, HGBParamGrid, KNNParamGrid
from ktools.hyperparameter_optimization.optuna_hyperparameter_optimizer import OptunaHyperparameterOptimizer
from ktools.modelling.models.catboost_model import CatBoostModel
from ktools.modelling.models.hgb_model import HGBModel
from ktools.modelling.models.knn_model import KNNModel
from ktools.modelling.models.lgbm_model import LGBMModel
from ktools.modelling.models.xgb_model import XGBoostModel
from ktools.preprocessing.basic_feature_transformers import FillNullValues
from ktools.utils.data_science_pipeline_settings import DataSciencePipelineSettings



class TestOptunaHyperparameterOptimizer(unittest.TestCase):

    def setUp(self):
        train_csv_path = "/Users/yuwei-1/Documents/projects/Kaggle-tools/data/used_car_prices/train.csv"
        test_csv_path = "/Users/yuwei-1/Documents/projects/Kaggle-tools/data/used_car_prices/test.csv"
        target_col_name = "price"

        settings = DataSciencePipelineSettings(train_csv_path,
                                                test_csv_path,
                                                target_col_name)
        
        settings = FillNullValues.transform(settings)
        train_df, test_df = settings.update()

        train_df[settings.categorical_col_names] = train_df[settings.categorical_col_names].astype("category")
        self.cat_cols = settings.categorical_col_names
        self.X, self.y = train_df.drop(columns=target_col_name), train_df[target_col_name]

    def test_reproducibility_lgbm(self):

        kf = KFold(n_splits=5, shuffle=True, random_state=123)
            
        lgbm = LGBMModel
        optimizer = OptunaHyperparameterOptimizer(self.X,
                                                  self.y,
                                                  lgbm,
                                                  BaseLGBMParamGrid(),
                                                  kf,
                                                  root_mean_squared_error,
                                                  n_trials=2,
                                                  direction="minimize")
        best_params = optimizer.optimize()
        model = LGBMModel(**best_params)

        cv_scores, oof, model_list = CrossValidationExecutor(model,
                                                            root_mean_squared_error,
                                                            kf,
                                                            use_test_as_valid=True
                                                            ).run(self.X, self.y)
        
        expected_params = {'num_leaves': 230, 
                           'max_depth': 0, 
                           'learning_rate': 0.3972192845559754, 
                           'num_boost_round': 91, 
                           'subsample': 0.6658087572237882, 
                           'colsample_bytree': 0.9352641520216103, 
                           'reg_alpha': 1.5549456825135817e-06, 
                           'reg_lambda': 0.0005586891596157437, 
                           'min_data_in_leaf': 90, 
                           'bagging_freq': 1, 
                           'min_child_weight': 1.1736072899798689, 
                           'cat_smooth': 12.58018028437027,
                           }
        
        self.assertEqual(expected_params, best_params)
        self.assertEqual(cv_scores[0], 73396.1853451514)


    def test_reproducibility_xgb(self):

        kf = KFold(n_splits=5, shuffle=True, random_state=123)
            
        xgb = XGBoostModel
        optimizer = OptunaHyperparameterOptimizer(self.X,
                                                  self.y,
                                                  xgb,
                                                  BaseXGBoostParamGrid(),
                                                  kf,
                                                  root_mean_squared_error,
                                                  n_trials=2,
                                                  direction="minimize")
        best_params = optimizer.optimize()
        model = XGBoostModel(**best_params)

        cv_scores, oof, model_list = CrossValidationExecutor(model,
                                                            root_mean_squared_error,
                                                            kf,
                                                            use_test_as_valid=True
                                                            ).run(self.X, self.y)
        
        expected_params = {'max_bin': 230, 
                           'learning_rate': 0.0010347984019709346, 
                           'max_depth': 41, 
                           'num_boost_round': 91, 
                           'gamma': 3.316175144475764, 
                           'min_child_weight': 40.886968543614735, 
                           'subsample': 0.5136939444870281, 
                           'colsample_bytree': 0.6962264460513162, 
                           'colsample_bylevel': 0.9497408696893027, 
                           'colsample_bynode': 0.5061555847824688, 
                           'reg_alpha': 0.05594697353802973, 
                           'reg_lambda': 0.00706161485759505, 
                           'max_cat_threshold': 2, 
                           'grow_policy': 'depthwise'}
        
        self.assertEqual(expected_params, best_params)
        self.assertEqual(cv_scores[0], 78019.27348376985)

    def test_reproducibility_cat(self):

        kf = KFold(n_splits=5, shuffle=True, random_state=123)
            
        cat = CatBoostModel
        optimizer = OptunaHyperparameterOptimizer(self.X,
                                                  self.y,
                                                  cat,
                                                  BaseCatBoostParamGrid(),
                                                  kf,
                                                  root_mean_squared_error,
                                                  n_trials=2,
                                                  direction="minimize")
        best_params = optimizer.optimize()
        model = CatBoostModel(**best_params)

        cv_scores, oof, model_list = CrossValidationExecutor(model,
                                                            root_mean_squared_error,
                                                            kf,
                                                            use_test_as_valid=True
                                                            ).run(self.X, self.y)
        
        expected_params = {'max_bin': 236, 
                           'learning_rate': 0.011111557266579378, 
                           'depth': 15, 
                           'iterations': 60, 
                           'bagging_temperature': 4.585782568882233, 
                           'subsample': 0.968692607913849, 
                           'colsample_bylevel': 0.5032824282936514, 
                           'min_data_in_leaf': 13.998024718435557, 
                           'l2_leaf_reg': 5.543304539930465, 
                           'grow_policy': 'Lossguide', 
                           'leaf_estimation_iterations': 4, 
                           'random_strength': 5.85658096122518, 
                           'leaf_estimation_method': 'Newton'}
        
        self.assertEqual(expected_params, best_params)
        self.assertEqual(cv_scores[0], 74657.56211249501)


    def test_reproducibility_knn(self):

        kf = KFold(n_splits=5, shuffle=True, random_state=123)
            
        cat = KNNModel

        cat_cols = self.cat_cols
        class _KNNParamGrid(KNNParamGrid):
            
            def get(self, trial : optuna.Trial):
                base_params = super().get(trial)
                params = {
                    "categorical_features" : cat_cols,
                }
                params.update(base_params)
                return params

        optimizer = OptunaHyperparameterOptimizer(self.X,
                                                  self.y,
                                                  cat,
                                                  _KNNParamGrid(),
                                                  kf,
                                                  root_mean_squared_error,
                                                  n_trials=2,
                                                  direction="minimize")
        best_params = optimizer.optimize()
        best_params.update({"categorical_features" : cat_cols})
        model = KNNModel(**best_params)

        cv_scores, oof, model_list = CrossValidationExecutor(model,
                                                            root_mean_squared_error,
                                                            kf,
                                                            use_test_as_valid=True
                                                            ).run(self.X, self.y)
        
        expected_params = {'smooth': 823.4599929986363, 
                           'weights': 'uniform', 
                           'n_neighbors': 1712, 
                           'min_max_scaling': True,
                           'categorical_features' : cat_cols}
        
        self.assertEqual(expected_params, best_params)
        self.assertEqual(cv_scores[0], 74032.38900824237)


    def test_reproducibility_hgb(self):

        kf = KFold(n_splits=5, shuffle=True, random_state=123)
            
        cat = HGBModel

        # cat_cols = self.cat_cols
        optimizer = OptunaHyperparameterOptimizer(self.X,
                                                  self.y,
                                                  cat,
                                                  HGBParamGrid(),
                                                  kf,
                                                  root_mean_squared_error,
                                                  n_trials=2,
                                                  direction="minimize")
        best_params = optimizer.optimize()

        print(best_params)
        model = HGBModel(**best_params)

        cv_scores, oof, model_list = CrossValidationExecutor(model,
                                                            root_mean_squared_error,
                                                            kf,
                                                            use_test_as_valid=True
                                                            ).run(self.X, self.y)
        
        expected_params = {'max_bins': 236, 
                           'learning_rate': 0.011111557266579378, 
                           'max_depth': 43, 
                           'max_leaf_nodes': 2, 
                           'min_samples_leaf': 167, 
                           'num_boost_round': 941, 
                           'validation_fraction': 0.05098472848809544, 
                           'early_stopping_rounds': 5, 
                           'l2_regularization': 5.543304539930465, 
                           'max_features': 0.5172806957113423, 
                           'interaction_cst': 'no_interactions', 
                           'tol': 4.989999866609151e-05, 
                           'smooth': 183.303203732075}
        
        self.assertEqual(expected_params, best_params)
        self.assertEqual(cv_scores[0], 73121.7989656374)