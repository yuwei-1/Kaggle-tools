import optuna
import joblib
import pandas as pd
import numpy as np
from optuna.samplers import TPESampler
from typing import *
from ktools.fitting.i_sklearn_model import ISklearnModel
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
                 verbose=False
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

    def optimize(self, 
                 inital_parameters : Dict[str, float] = None,
                 initial_distribution : Dict[str, Any] = None,
                 timeout : int = 3600
                 ):
        if self._verbose:
            print("Starting Optuna trials...........................")
        sampler = TPESampler(seed=int(self._n_trials*self._explore_fraction))
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
        oof_predictions = np.zeros(self._y_train.shape[0])
        for train_index, val_index in self._kfold_object.split(self._X_train, self._y_train):
            X_train_fold, X_val_fold = self._X_train.iloc[train_index], self._X_train.iloc[val_index]
            y_train_fold, _ = self._y_train.iloc[train_index], self._y_train.iloc[val_index]

            model = self.model(**parameters)
            
            model.fit(X_train_fold, 
                      y_train_fold)

            oof_predictions[val_index] = model.predict(X_val_fold)

        return self._metric(self._y_train.to_numpy().squeeze(), oof_predictions)