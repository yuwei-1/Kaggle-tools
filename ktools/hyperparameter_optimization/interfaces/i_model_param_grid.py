import optuna
from abc import ABC, abstractmethod

class IModelParamGrid:
    @staticmethod
    @abstractmethod
    def get(trial : optuna.Trial):
        pass