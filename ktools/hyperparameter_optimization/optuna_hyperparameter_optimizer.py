import optuna
from ktools.hyperparameter_optimization.interfaces.i_hyperparameter_optimizer import IHyperparameterOptimizer


class OptunaHyperparameterOptimizer(IHyperparameterOptimizer):

    def __init__(self, 
                 objective : callable,
                 direction='maximize',
                 n_trials=100,
                 study_name="ml_experiment",
                 verbose=False
                 ) -> None:
        super().__init__()
        self._objective = objective
        self._direction = direction
        self._n_trials = n_trials
        self._study_name = study_name
        self._verbose = verbose

    def optimize(self):
        study = optuna.create_study(study_name=self._study_name, direction=self._direction)
        study.optimize(self._objective, n_trials=self._n_trials)
        optimal_params = study.best_params
        if self._verbose:
            print(f"Best Parameters: {optimal_params}")
        return optimal_params