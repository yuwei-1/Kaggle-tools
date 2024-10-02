import optuna
import joblib
import pandas as pd
import numpy as np
from optuna.samplers import TPESampler
from typing import *
from ktools.fitting.cross_validation_executor import CrossValidationExecutor
from ktools.modelling.Interfaces.i_sklearn_model import ISklearnModel
from ktools.hyperparameter_optimization.i_hyperparameter_optimizer import IHyperparameterOptimizer
from ktools.hyperparameter_optimization.i_model_param_grid import IModelParamGrid
from ktools.hyperparameter_optimization.i_sklearn_kfold_object import ISklearnKFoldObject


class OptunaHyperparameterOptimizer(IHyperparameterOptimizer):

    def __init__(self,
                 X_train : pd.DataFrame,
                 y_train : pd.DataFrame,
                 model : ISklearnModel,
                 param_grid_getter : IModelParamGrid,
                 kfold_object : ISklearnKFoldObject,
                 metric : callable,
                 direction : str = 'maximize',
                 n_trials : int = 100,
                 study_name : str = "ml_experiment",
                 explore_fraction : float = 0.1,
                 verbose=False,
                 random_state=42
                 ) -> None:
        
        super().__init__()
        self._X_train = X_train
        self._y_train = y_train
        self.model = model
        self._metric = metric
        self._param_grid_getter = param_grid_getter
        self._kfold_object = kfold_object
        self._direction = direction
        self._n_trials = n_trials
        self._study_name = study_name
        self._explore_fraction = explore_fraction
        self._verbose = verbose
        self._random_state = random_state

    def optimize(self, 
                 inital_parameters : Dict[str, float] = None,
                 initial_distribution : Dict[str, Any] = None,
                 timeout : int = 3600
                 ):
        if self._verbose:
            print("#"*100)
            print("Starting Optuna Optimizer")
            print("#"*100)

        sampler = TPESampler(n_startup_trials=int(self._n_trials*self._explore_fraction),
                             seed=self._random_state)
        study = optuna.create_study(sampler=sampler,
                                    study_name=self._study_name, 
                                    direction=self._direction)
        
        if inital_parameters is not None:
            fixed_trial = optuna.trial.FixedTrial(inital_parameters)
            study.add_trial(optuna.create_trial(
                            params=inital_parameters,
                            distributions=initial_distribution,
                            value=self._objective(fixed_trial)
            ))
        study.optimize(self._objective, n_trials=self._n_trials, timeout=timeout)
        # joblib.dump(study, "/kaggle/working/study.pkl")
        optimal_params = study.best_params
        return optimal_params
    
    def _objective(self, trial : optuna.Trial):
        parameters = self._param_grid_getter.get(trial)

        cv_scores, oof, model_list = CrossValidationExecutor(self.model(**parameters),
                                                             self._metric,
                                                             self._kfold_object,
                                                             use_test_as_valid=True
                                                             ).run(self._X_train, self._y_train)

        return cv_scores[0]