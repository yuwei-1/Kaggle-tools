import optuna
from ktools.hyperparameter_optimization.i_model_param_grid import IModelParamGrid

class BaseLGBMParamGrid(IModelParamGrid):
    @staticmethod
    def get(trial : optuna.Trial):
        params = {
            # "boosting_type" : "gbdt",
            "num_leaves" : trial.suggest_int("num_leaves", 2, 500),
            "max_depth" : trial.suggest_int("max_depth", 0, 50),
            "learning_rate" : trial.suggest_float("learning_rate", 1e-2, 1.0, log=True),
            "num_boost_round" : trial.suggest_int("num_boost_round", 50, 1000),
            "subsample" : trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree" : trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha" : trial.suggest_float("reg_alpha", 1e-6, 10, log=True),
            "reg_lambda" : trial.suggest_float("reg_lambda", 1e-6, 10, log=True),
            "min_data_in_leaf" : trial.suggest_int("min_data_in_leaf", 1, 100),
            "bagging_freq" : trial.suggest_int("bagging_freq", 1, 5),
            "data_sample_strategy" : "bagging",
            "min_child_weight": trial.suggest_float("min_child_weight", 1e-4, 100, log=True),
            "cat_smooth": trial.suggest_float("cat_smooth", 1, 100, log=True)
            }
        return params


class LGBMGBDTParamGrid(BaseLGBMParamGrid):

    def get(self, trial : optuna.Trial):
        base_params = super().get(trial)
        params = {
            "boosting_type" : "gbdt",
        }
        params.update(base_params)
        return params
    
class LGBMDARTParamGrid(BaseLGBMParamGrid):

    def get(self, trial : optuna.Trial):
        base_params = super().get(trial)
        params = {
            "boosting_type" : "dart",
        }
        params.update(base_params)
        return params
    
class LGBMRFParamGrid(BaseLGBMParamGrid):

    def get(self, trial : optuna.Trial):
        base_params = super().get(trial)
        params = {
            "boosting_type" : "rf",
        }
        params.update(base_params)
        return params
    

class BaseXGBoostParamGrid(IModelParamGrid):
    @staticmethod
    def get(trial : optuna.Trial):
        params = {
            # "booster": trial.suggest_categorical("booster", ["gbtree", "dart"]),
            "max_bin" : trial.suggest_int("max_bin", 2, 500),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 50),
            "num_boost_round" : trial.suggest_int("num_boost_round", 50, 1000),
            "gamma" : trial.suggest_float("gamma", 0, 10),
            "min_child_weight" : trial.suggest_float("min_child_weight", 0.1, 100, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.5, 1.0),
            "colsample_bynode": trial.suggest_float("colsample_bynode", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-6, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-6, 10.0, log=True),
            "max_cat_threshold" : trial.suggest_int("max_cat_threshold", 1, 1000, log=True),
            "sampling_method" : "uniform",
            "grow_policy" : trial.suggest_categorical("grow_policy", ["lossguide", "depthwise"]),
        }
        return params
    
class XGBoostGBTree(BaseXGBoostParamGrid):

    def get(self, trial : optuna.Trial):
        base_params = super().get(trial)
        params = {
            "booster" : "gbtree",
        }
        params.update(base_params)
        return params
    
class XGBoostGBTreeLinear(BaseXGBoostParamGrid):

    def get(self, trial : optuna.Trial):
        base_params = super().get(trial)
        params = {
            "booster" : "gblinear",
        }
        params.update(base_params)
        return params
    
class XGBoostDART(BaseXGBoostParamGrid):

    def get(self, trial : optuna.Trial):
        base_params = super().get(trial)
        params = {
            "booster" : "dart",
        }
        params.update(base_params)
        return params
    

class BaseCatBoostParamGrid(IModelParamGrid):
    @staticmethod
    def get(trial : optuna.Trial):
        params = {
            # "booster": trial.suggest_categorical("booster", ["gbtree", "dart"]),
            "max_bin" : trial.suggest_int("max_bin", 2, 500),
            "learning_rate": trial.suggest_float("learning_rate", 1e-2, 0.3, log=True),
            "depth": trial.suggest_int("depth", 3, 16),
            "iterations" : trial.suggest_int("iterations", 50, 1000),
            "bagging_temperature" : trial.suggest_float("bagging_temperature", 1, 100, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bylevel" : trial.suggest_float("colsample_bylevel", 0.5, 1.0),
            "min_data_in_leaf" : trial.suggest_float("min_data_in_leaf", 1, 1000, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-6, 10.0, log=True),
            "grow_policy" : trial.suggest_categorical("grow_policy", ["Lossguide", "Depthwise", "SymmetricTree"]),
            "leaf_estimation_iterations" : trial.suggest_int("leaf_estimation_iterations", 1, 5),
            "random_strength" : trial.suggest_float("random_strength", 1, 10),
            "leaf_estimation_method" : trial.suggest_categorical("leaf_estimation_method", ["Newton", "Gradient"]),
            "use_best_model" : True
        }
        return params
    

class KNNParamGrid(IModelParamGrid):
    @staticmethod
    def get(trial : optuna.Trial):
        params = {
            "smooth" : trial.suggest_float("smooth", 1e2, 1e4, log=True),
            "weights" : trial.suggest_categorical("weights", ["uniform", "distance"]),
            "n_neighbors" : trial.suggest_int("n_neighbors", 2, 1e4, log=True),
            "min_max_scaling" : trial.suggest_categorical("min_max_scaling", [True, False])
        }
        return params
    

class SVMParamGrid(IModelParamGrid):
    @staticmethod
    def get(trial : optuna.Trial):
        params = {
            "smooth" : trial.suggest_float("smooth", 1e2, 1e4, log=True),
            "weights" : trial.suggest_categorical("weights", ["uniform", "distance"]),
            "standard_scaling" : trial.suggest_categorical("standard_scaling", [True, False]),
            "kernel" : trial.suggest_categorical("kernel", ['linear', 'poly', 'rbf', 'sigmoid']),
            "degree" : trial.suggest_int("degree", 1, 6),
            "gamma" : trial.suggest_float("gamma", 1e-6, 100, log=True),
            "tol" : trial.suggest_float("tol", 1e-3, 1),
            "C" : trial.suggest_float("C", 1e-6, 10, log=True),
            "shrinking" : trial.suggest_categorical("shrinking", [True, False]),
            "max_iter" : trial.suggest_int("max_iter", 1e2, 1e5, log=True)
        }
        return params
    

class HGBParamGrid(IModelParamGrid):
    @staticmethod
    def get(trial : optuna.Trial):
        params = {
            "max_bins" : trial.suggest_int("max_bins", 2, 500),
            "learning_rate": trial.suggest_float("learning_rate", 1e-2, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 50),
            "max_leaf_nodes" : trial.suggest_int("max_leaf_nodes", 2, 400, log=True),
            "min_samples_leaf" : trial.suggest_int("min_samples_leaf", 2, 500),
            "num_boost_round" : trial.suggest_int("num_boost_round", 50, 1000),
            "validation_fraction" : trial.suggest_float("validation_fraction", 0.05, 0.2),
            "early_stopping_rounds" : trial.suggest_int("early_stopping_rounds", 1, 200, log=True),
            "l2_regularization": trial.suggest_float("l2_regularization", 1e-6, 10.0, log=True),
            "max_features": trial.suggest_float("max_features", 0.5, 1.0),
            "interaction_cst" : trial.suggest_categorical("interaction_cst", ["pairwise", "no_interactions"]),
            "tol": trial.suggest_float("tol", 1e-7, 1e-2, log=True), 
            "smooth" : trial.suggest_float("smooth", 1e2, 1e4, log=True),           
        }
        return params