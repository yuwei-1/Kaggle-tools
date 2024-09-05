import unittest
import pandas as pd
import numpy as np
import optuna
from sklearn.metrics import r2_score
from sklearn.linear_model import Lasso
from sklearn.model_selection import KFold
from ktools.hyperparameter_optimization.i_model_param_grid import IModelParamGrid
from ktools.hyperparameter_optimization.optuna_hyperparameter_optimizer import OptunaHyperparameterOptimizer



class TestOptunaHyperparameterOptimizer(unittest.TestCase):

    def setUp(self):
        self.x_train = pd.DataFrame(np.arange(1, 10, 1))
        self.y_train = self.x_train**2

        class LassoParamGrid(IModelParamGrid):
            @staticmethod
            def get(trial : optuna.Trial):
                params = {
                'alpha': trial.suggest_float('alpha', 1e-2, 1e3, log=True),
                }
                return params
        
        self.param_getter = LassoParamGrid()
        self.model = Lasso

        self.kf = KFold(n_splits=2, shuffle=True, random_state=123)
        

    def test_optimize(self):
        optimizer = OptunaHyperparameterOptimizer(self.x_train,
                                                  self.y_train,
                                                  self.model,
                                                  self.param_getter,
                                                  self.kf,
                                                  r2_score,
                                                  n_trials=100)
        best_params = optimizer.optimize()

        self.assertTrue({'alpha': 0.01004845284649519} == best_params)