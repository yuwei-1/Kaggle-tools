import optuna
from ktools.hyperparameter_optimization.interfaces.i_model_param_grid import IModelParamGrid

class BaseLGBMParamGrid(IModelParamGrid):
    @staticmethod
    def get(trial : optuna.Trial):
        params = {
            # "boosting_type" : "gbdt",
            "num_leaves" : trial.suggest_int("num_leaves", 2, 500),
            "max_depth" : trial.suggest_int("max_depth", -1, 20),
            "learning_rate" : trial.suggest_float("learning_rate", 1e-3, 1.0, log=True),
            "n_estimators" : trial.suggest_int("n_estimators", 50, 1500),
            "subsample" : trial.suggest_float("subsample", 0.1, 1.0),
            "colsample_bytree" : trial.suggest_float("colsample_bytree", 0.1, 1.0),
            "reg_alpha" : trial.suggest_float("reg_alpha", 1e-8, 10, log=True),
            "reg_lambda" : trial.suggest_float("reg_lambda", 1e-8, 10, log=True),
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