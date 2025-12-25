from typing import Any, Callable, Dict, Tuple, Type
import optuna
from optuna.samplers import TPESampler
from ktools import logger
from ktools.utils.loader import TrialSampler, load_optuna_grid


class OptunaHyperparameterOptimizer:
    """
    Hyperparameter optimizer using Optuna's TPE sampler.

    Args:
        model_type: Type of model to optimize (e.g., "catboost", "lightgbm").
        grid_yaml_path: Path to YAML file containing parameter search space.
        timeout: Maximum optimization time in seconds.
        direction: Optimization direction ("maximize" or "minimize").
        n_trials: Number of trials to run.
        study_name: Name for the Optuna study.
        explore_fraction: Fraction of trials for exploration phase.
        save_study: Whether to persist the study to SQLite.
        load_if_exists: Whether to resume an existing study with the same name.
        catch_exceptions: Tuple of exception types to catch during optimization.
        verbose: Whether to log progress information.
        random_state: Random seed for reproducibility.
    """

    def __init__(
        self,
        model_type: str,
        grid_yaml_path: str,
        extra_samplers: Dict[str, TrialSampler] | None = None,
        timeout: int = 3600,
        direction: str = "maximize",
        n_trials: int = 100,
        study_name: str = "ml_experiment",
        explore_fraction: float = 0.1,
        save_study: bool = False,
        load_if_exists: bool = True,
        catch_exceptions: Tuple[Type[Exception], ...] = (),
        verbose: bool = False,
        random_state: int = 42,
    ) -> None:
        self._param_space_builder = load_optuna_grid(
            grid_yaml_path, model_type, extra_samplers=extra_samplers
        )
        self._timeout = timeout
        self._direction = direction
        self._n_trials = n_trials
        self._study_name = study_name
        self._explore_fraction = explore_fraction
        self._save_study = save_study
        self._load_if_exists = load_if_exists
        self._catch_exceptions = catch_exceptions
        self._verbose = verbose
        self._random_state = random_state
        self.study: optuna.Study | None = None

    def optimize(
        self,
        *args: Any,
        tunable_func: Callable[..., float],
    ) -> dict[str, Any]:
        """
        Run hyperparameter optimization.

        Args:
            *args: Positional arguments passed to tunable_func.
            tunable_func: Function that takes (*args, **hyperparameters) and returns a score.

        Returns:
            Dictionary of best hyperparameters found.
        """
        if self._verbose:
            logger.info("Starting Optuna Optimizer")

        sampler = TPESampler(
            n_startup_trials=int(self._n_trials * self._explore_fraction),
            seed=self._random_state,
        )

        storage_name = f"sqlite:///{self._study_name}.db" if self._save_study else None

        self.study = optuna.create_study(
            sampler=sampler,
            study_name=self._study_name,
            direction=self._direction,
            storage=storage_name,
            load_if_exists=self._load_if_exists,
        )

        def objective(trial: optuna.Trial) -> float:
            parameters = self._param_space_builder(trial)
            return tunable_func(*args, **parameters)

        self.study.optimize(
            objective,
            n_trials=self._n_trials,
            timeout=self._timeout,
            catch=self._catch_exceptions,
        )

        return self.study.best_params
