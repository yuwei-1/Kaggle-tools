from abc import ABC, abstractmethod
from ktools.modelling.Interfaces.i_sklearn_model import ISklearnModel


class IModelWrapper(ABC):
    
    def __init__(self, model : ISklearnModel) -> None:
        self.model = model

    @abstractmethod
    def fit(self):
        pass

    @abstractmethod
    def predict(self):
        pass