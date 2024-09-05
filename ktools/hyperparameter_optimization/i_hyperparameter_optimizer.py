from abc import ABC, abstractmethod


class IHyperparameterOptimizer(ABC):

    @abstractmethod
    def optimize(self):
        pass