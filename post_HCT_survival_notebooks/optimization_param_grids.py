import optuna


lgb_regression_base_params = {'objective' : 'regression', 'metric' : 'rmse', "num_boost_round" : 10000, "early_stopping_rounds" : 100,}
xgb_regression_base_params = {"objective": "reg:squarederror", "eval_metric": "rmse", "num_boost_round" : 10000, "early_stopping_rounds" : 100}
xgb_cox_base_params = {"objective": "survival:cox", "eval_metric": "cox-nloglik", "num_boost_round" : 10000, "early_stopping_rounds" : 100}
cat_cox_base_params = {'loss_function' : 'Cox', 'eval_metric' : 'Cox', "num_boost_round" : 10000, "early_stopping_rounds" : 100}
cat_regression_base_params = {"loss_function" : "RMSE", 'eval_metric': 'RMSE', "num_boost_round" : 10000, "early_stopping_rounds" : 100}


class BaseLGBMParamGrid():
    @staticmethod
    def get(trial : optuna.Trial):
        params = {
            "num_leaves" : trial.suggest_int("num_leaves", 2, 500),
            "max_depth" : trial.suggest_int("max_depth", 0, 50),
            "learning_rate" : trial.suggest_float("learning_rate", 1e-2, 1.0, log=True),
            "subsample" : trial.suggest_float("subsample", 0.7, 1.0),
            "colsample_bytree" : trial.suggest_float("colsample_bytree", 0.7, 1.0),
            "reg_alpha" : trial.suggest_float("reg_alpha", 1e-6, 10, log=True),
            "reg_lambda" : trial.suggest_float("reg_lambda", 1e-6, 10, log=True),
            "min_data_in_leaf" : trial.suggest_int("min_data_in_leaf", 1, 100),
            "feature_fraction" : trial.suggest_float("feature_fraction", 0.5, 1.0),
            # "bagging_fraction" : trial.suggest_float("bagging_fraction", 0.5, 1.0),
            # "bagging_freq" : trial.suggest_int("bagging_freq", 1, 5),
            "max_bin" : trial.suggest_int("max_bin", 50, 5000, log=True),
            'min_child_weight': trial.suggest_float('min_child_weight', 1e-4, 100, log=True),
            # "scale_pos_weight" : trial.suggest_float("scale_pos_weight", 1, 1000, log=True),
            'cat_smooth': trial.suggest_float('cat_smooth', 1, 100, log=True)}
        return params


class LGBMGBDTParamGrid(BaseLGBMParamGrid):
    def get(self, trial : optuna.Trial):
        base_params = super().get(trial)
        params = {
            "boosting_type" : "gbdt",
            "data_sample_strategy" : "bagging"
        }
        params.update(base_params)
        return params


class LGBMGBDTGossParamGrid(BaseLGBMParamGrid):
    def get(self, trial : optuna.Trial):
        base_params = super().get(trial)
        params = {
            "boosting_type" : "gbdt",
            "data_sample_strategy" : "goss"
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
            "num_boost_round" : 10000
        }
        params.update(base_params)
        return params

    

class BaseXGBoostParamGrid():
    @staticmethod
    def get(trial : optuna.Trial):
        params = {
            "max_bin" : trial.suggest_int("max_bin", 50, 5000, log=True),
            # "max_bin" : 10000,
            "learning_rate": trial.suggest_float("learning_rate", 1e-2, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 50),
            "gamma" : trial.suggest_float("gamma", 0, 10),
            "min_child_weight" : trial.suggest_float("min_child_weight", 0.1, 100, log=True),
            "subsample": trial.suggest_float("subsample", 0.8, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.8, 1.0),
            "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.8, 1.0),
            "colsample_bynode": trial.suggest_float("colsample_bynode", 0.8, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-6, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-6, 10.0, log=True),
            "max_cat_threshold" : trial.suggest_int("max_cat_threshold", 1, 1000, log=True),
            "max_cat_to_onehot" : trial.suggest_int("max_cat_to_onehot", 1, 1000, log=True)
            # "scale_pos_weight" : trial.suggest_float("scale_pos_weight", 1, 1000, log=True),
        }
        return params

class XGBoostGBTreeLossguide(BaseXGBoostParamGrid):
    def get(self, trial : optuna.Trial):
        base_params = super().get(trial)
        params = {
            "booster" : "gbtree",
            "grow_policy" : "lossguide"
        }
        params.update(base_params)
        return params

class XGBoostGBTreeDepthwise(BaseXGBoostParamGrid):
    def get(self, trial : optuna.Trial):
        base_params = super().get(trial)
        params = {
            "booster" : "gbtree",
            "grow_policy" : "depthwise"
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

    

class BaseCatBoostParamGrid():
    @staticmethod
    def get(trial : optuna.Trial):
        params = {
            # "booster": trial.suggest_categorical("booster", ["gbtree", "dart"]),
            "max_bin" : trial.suggest_int("max_bin", 2, 5000, log=True),
            # "max_bin" : 5000,
            "learning_rate": trial.suggest_float("learning_rate", 0.1, 0.3, log=True),
            "depth": trial.suggest_int("depth", 3, 16),
#             trial.suggest_int("num_boost_round", 50, 5000, log=True),
            "bagging_temperature" : trial.suggest_float("bagging_temperature", 0.1, 100, log=True),
            "subsample": trial.suggest_float("subsample", 0.7, 1.0),
            "colsample_bylevel" : trial.suggest_float("colsample_bylevel", 0.7, 1.0),
            "min_data_in_leaf" : trial.suggest_float("min_data_in_leaf", 1, 1000, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-6, 10.0, log=True),
#             "grow_policy" : trial.suggest_categorical("grow_policy", ["Lossguide", "Depthwise", "SymmetricTree"]),
            "leaf_estimation_iterations" : trial.suggest_int("leaf_estimation_iterations", 1, 5),
            # "scale_pos_weight" : trial.suggest_float("scale_pos_weight", 1, 1000, log=True),
            "random_strength" : trial.suggest_float("random_strength", 0.1, 10),
            "leaf_estimation_method" : trial.suggest_categorical("leaf_estimation_method", ["Newton", "Gradient"]),
        }
        return params

class GPUCatBoostParamGrid():
    @staticmethod
    def get(trial : optuna.Trial):
        params = {
            "max_bin" : trial.suggest_int("max_bin", 2, 5000, log=True),
            "learning_rate": trial.suggest_float("learning_rate", 1e-2, 0.3, log=True),
            "depth": trial.suggest_int("depth", 3, 16),
            "bagging_temperature" : trial.suggest_float("bagging_temperature", 0.1, 100, log=True),
            # "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            # "colsample_bylevel" : trial.suggest_float("colsample_bylevel", 0.5, 1.0),
            "min_data_in_leaf" : trial.suggest_float("min_data_in_leaf", 1, 1000, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-6, 10.0, log=True),
            "leaf_estimation_iterations" : trial.suggest_int("leaf_estimation_iterations", 1, 5),
            # "scale_pos_weight" : trial.suggest_float("scale_pos_weight", 1, 1000, log=True),
            # "random_strength" : trial.suggest_float("random_strength", 0.1, 10),
            "leaf_estimation_method" : trial.suggest_categorical("leaf_estimation_method", ["Newton", "Gradient"]),
            # 'loss_function' : "Logloss",
            # 'eval_metric': 'AUC',
            'task_type' : "GPU",
            'devices' : '0,1',
            "bootstrap_type" : "Bayesian"
        }
        return params

class CatBoostDepthWise(BaseCatBoostParamGrid):
    def get(self, trial : optuna.Trial):
        base_params = super().get(trial)
        params = {
            "grow_policy" : "Depthwise",
        }
        params.update(base_params)
        return params

    
class CatBoostLossGuide(BaseCatBoostParamGrid):
    def get(self, trial : optuna.Trial):
        base_params = super().get(trial)
        params = {
            "grow_policy" : "Lossguide",
        }
        params.update(base_params)
        return params

    
class CatBoostSymmetricTree(BaseCatBoostParamGrid):
    def get(self, trial : optuna.Trial):
        base_params = super().get(trial)
        params = {
            "grow_policy" : "SymmetricTree",
        }
        params.update(base_params)
        return params


class CatBoostRegion(BaseCatBoostParamGrid):
    def get(self, trial : optuna.Trial):
        base_params = super().get(trial)
        params = {
            "grow_policy" : "Region",
            'task_type' : "GPU",
        }
        params.update(base_params)
        return params

    
class HGBParamGrid():
    @staticmethod
    def get(trial : optuna.Trial):
        params = {
            "max_bins" : trial.suggest_int("max_bins", 2, 255),
            "learning_rate": trial.suggest_float("learning_rate", 1e-2, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 50),
            "max_leaf_nodes" : trial.suggest_int("max_leaf_nodes", 2, 400, log=True),
            "min_samples_leaf" : trial.suggest_int("min_samples_leaf", 2, 500),
            # "num_boost_round" : trial.suggest_int("num_boost_round", 50, 3000),
            "num_boost_round" : 3000,
            "early_stopping_rounds" : 20,
            "validation_fraction" : trial.suggest_float("validation_fraction", 0.05, 0.2),
            # "early_stopping_rounds" : trial.suggest_int("early_stopping_rounds", 1, 200, log=True),
            "l2_regularization": trial.suggest_float("l2_regularization", 1e-6, 10.0, log=True),
            "max_features": trial.suggest_float("max_features", 0.5, 1.0),
            "interaction_cst" : trial.suggest_categorical("interaction_cst", ["pairwise", "no_interactions"]),
            "tol": trial.suggest_float("tol", 1e-7, 1e-2, log=True), 
            "smooth" : trial.suggest_float("smooth", 1e2, 1e4, log=True),           
        }
        return params


class YDFParamGrid():
    @staticmethod
    def get(trial : optuna.Trial):
        params = {
            "categorical_algorithm" : trial.suggest_categorical("categorical_algorithm", ["CART", "RANDOM"]),
            "categorical_set_split_min_item_frequency" : trial.suggest_int("categorical_set_split_min_item_frequency", 1, 200),
            # "goss_alpha" : trial.suggest_float("goss_alpha", 0, 1),
            # "goss_beta" : trial.suggest_float("goss_beta", 0, 1),
            # "honest" : trial.suggest_categorical("honest", [True, False]),
            "l1_regularization" : trial.suggest_float("l1_regularization", 1e-4, 1e2, log=True),
            "l2_categorical_regularization" : trial.suggest_float("l2_categorical_regularization", 1e-4, 1e2, log=True),
            "l2_regularization" : trial.suggest_float("l2_regularization", 1e-4, 1e2, log=True),
            "max_depth" : trial.suggest_int("max_depth", -1, 300),
            "max_num_nodes" : trial.suggest_int("max_num_nodes", -1, 200),
            "min_examples" : trial.suggest_int("min_examples", 1, 1e3, log=True),
            "num_boost_round" : 3000,
            "task" : "REGRESSION",
            "loss" : "SQUARED_ERROR"
        }
        return params
    

class YDFLocalParamGrid(YDFParamGrid):
    def get(self, trial : optuna.Trial):
        base_params = super().get(trial)
        params = {
            "growing_strategy" : "LOCAL",
        }
        params.update(base_params)
        return params
    
class YDFBestGlobalParamGrid(YDFParamGrid):
    def get(self, trial : optuna.Trial):
        base_params = super().get(trial)
        params = {
            "growing_strategy" : "BEST_FIRST_GLOBAL",
            "subsample" : trial.suggest_float("subsample", 0.5, 1),
        }
        params.update(base_params)
        return params