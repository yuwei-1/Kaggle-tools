import optuna
import pandas as pd
from optuna.samplers import TPESampler
from typing import *
from catboost import CatBoostError
from ktools.fitting.cross_validation_executor import CrossValidationExecutor
from ktools.modelling.Interfaces.i_model_wrapper import IModelWrapper


class OptunaHyperparameterOptimizer:
    def __init__(
        self,
        X_train: pd.DataFrame,
        y_train: pd.DataFrame,
        model,
        param_grid_getter,
        kfold_object,
        metric: callable,
        fixed_model_params={},
        model_wrapper: IModelWrapper = None,
        training_features: Union[List[str], None] = None,
        direction: str = "maximize",
        n_trials: int = 100,
        study_name: str = "ml_experiment",
        explore_fraction: float = 0.1,
        cross_validation_run_kwargs={},
        save_study: bool = False,
        verbose=False,
        random_state=42,
    ) -> None:
        super().__init__()
        self._X_train = X_train
        self._y_train = y_train
        self.model = model
        self._metric = metric
        self._fixed_model_params = fixed_model_params
        self._model_wrapper = model_wrapper
        self._training_features = training_features
        self._param_grid_getter = param_grid_getter
        self._kfold_object = kfold_object
        self._direction = direction
        self._n_trials = n_trials
        self._study_name = study_name
        self._explore_fraction = explore_fraction
        self._cross_validation_run_kwargs = cross_validation_run_kwargs
        self._save_study = save_study
        self._verbose = verbose
        self._random_state = random_state

    def optimize(
        self,
        inital_parameters: Dict[str, float] = None,
        initial_distribution: Dict[str, Any] = None,
        timeout: int = 3600,
    ):
        if self._verbose:
            print("#" * 100)
            print("Starting Optuna Optimizer")
            print("#" * 100)

        sampler = TPESampler(
            n_startup_trials=int(self._n_trials * self._explore_fraction),
            seed=self._random_state,
        )

        storage_name = (
            "sqlite:///{}.db".format(self._study_name) if self._save_study else None
        )
        self.study = study = optuna.create_study(
            sampler=sampler,
            study_name=self._study_name,
            direction=self._direction,
            storage=storage_name,
            load_if_exists=True,
        )

        if inital_parameters is not None:
            fixed_trial = optuna.trial.FixedTrial(inital_parameters)
            study.add_trial(
                optuna.create_trial(
                    params=inital_parameters,
                    distributions=initial_distribution,
                    value=self._objective(fixed_trial),
                )
            )
        study.optimize(
            self._objective,
            n_trials=self._n_trials,
            timeout=timeout,
            catch=(CatBoostError,),
        )
        optimal_params = study.best_params
        optimal_params.update(**self._fixed_model_params)
        return optimal_params

    def _objective(self, trial: optuna.Trial):
        parameters = self._param_grid_getter.get(trial)

        if self._model_wrapper is not None:
            model = self._model_wrapper.set_model(model)
            parameters = self._model_wrapper.take_params(parameters)

        model = self.model(**parameters, **self._fixed_model_params)

        cv_scores, oof, model_list, _ = CrossValidationExecutor(
            model,
            self._metric,
            self._kfold_object,
            training_features=self._training_features,
            use_test_as_valid=True,
        ).run(self._X_train, self._y_train, **self._cross_validation_run_kwargs)

        return cv_scores[0]
