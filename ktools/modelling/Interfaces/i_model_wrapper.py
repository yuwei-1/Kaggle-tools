from abc import ABC, abstractmethod
from typing import Any, Dict
from ktools.modelling.Interfaces.i_ktools_model import IKtoolsModel


class IModelWrapper(ABC):

    @abstractmethod
    def fit(self):
        pass

    @abstractmethod
    def predict(self):
        pass
    
    def set_model(self, model):
        self.model = model
        return self

    @abstractmethod
    def take_params(self, params : Dict[str, Any]):
        pass