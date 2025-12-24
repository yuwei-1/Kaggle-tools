import optuna
from optuna.samplers import TPESampler
from typing import *
from catboost import CatBoostError
from ktools.config.dataset import DatasetConfig
from ktools.fitting.cv_executor import CrossValidationExecutor
from ktools.fitting.pipe import ModelPipeline
from ktools.hyperopt.i_sklearn_kfold_object import ISklearnKFoldObject
from ktools.preprocessing.pipe import PreprocessingPipeline
from ktools.utils.loader import load_optuna_grid


class OptunaHyperparameterOptimizer:
    def __init__(
        self,
        model,
        grid_yaml_path: str,
        config: DatasetConfig,
        evaluation_metric: Callable,
        kfold_object: ISklearnKFoldObject,
        preprocessor: PreprocessingPipeline,
        timeout: int = 3600,
        model_type: str = "base",
        direction: str = "maximize",
        n_trials: int = 100,
        study_name: str = "ml_experiment",
        explore_fraction: float = 0.1,
        save_study: bool = False,
        verbose=False,
        random_state=42,
    ) -> None:
        super().__init__()
        self.model = model
        self._param_grid_getter = load_optuna_grid(grid_yaml_path, model_type)
        self.config = config
        self._evaluation_metric = evaluation_metric
        self._kfold_object = kfold_object
        self._preprocessor = preprocessor
        self._timeout = timeout
        self._direction = direction
        self._n_trials = n_trials
        self._study_name = study_name
        self._explore_fraction = explore_fraction
        self._save_study = save_study
        self._verbose = verbose
        self._random_state = random_state

    def optimize(
        self,
        *cv_args,
        **cv_kwargs,
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

        def objective(trial: optuna.Trial):
            parameters = self._param_grid_getter(trial)
            model = self.model(**parameters)

            cv_executor = CrossValidationExecutor(
                config=self.config,
                model_pipeline=ModelPipeline(
                    model=model,
                    config=self.config,
                    preprocessor=self._preprocessor,
                ),
                evaluation_metric=self._evaluation_metric,
                kfold_object=self._kfold_object,
            )
            score, _, _, _ = cv_executor.run(
                *cv_args,
                **cv_kwargs,
            )
            return score

        study.optimize(
            objective,
            n_trials=self._n_trials,
            timeout=self._timeout,
            catch=(CatBoostError,),
        )
        optimal_params = study.best_params
        return optimal_params
