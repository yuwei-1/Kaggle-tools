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
            "n_estimators" : trial.suggest_int("n_estimators", 50, 1000),
            "subsample" : trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree" : trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha" : trial.suggest_float("reg_alpha", 1e-6, 10, log=True),
            "reg_lambda" : trial.suggest_float("reg_lambda", 1e-6, 10, log=True),
            "min_data_in_leaf" : trial.suggest_int("min_data_in_leaf", 1, 100),
            "feature_fraction" : trial.suggest_float("feature_fraction", 0.5, 1.0),
            "bagging_fraction" : trial.suggest_float("bagging_fraction", 0.5, 1.0),
            "bagging_freq" : trial.suggest_int("bagging_freq", 1, 5),
            "data_sample_strategy" : "bagging",
            "min_child_weight": trial.suggest_float("min_child_weight", 1e-4, 100, log=True),
            "cat_smooth": trial.suggest_float("cat_smooth", 1, 100, log=True),
            "random_state" : 129,
            "verbose" : -1}
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
            "num_leaves" : trial.suggest_int("num_leaves", 2, 500),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 50),
            "n_estimators" : trial.suggest_int("n_estimators", 50, 1000),
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
            "random_state": 129,
            "verbosity": 0
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