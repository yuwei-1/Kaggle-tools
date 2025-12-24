import optuna
from abc import abstractmethod


class IModelParamGrid:
    @staticmethod
    @abstractmethod
    def get(trial: optuna.Trial):
        pass
