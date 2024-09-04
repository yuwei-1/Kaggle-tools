from abc import ABC, abstractmethod


class ISklearnModel(ABC):
    
    @abstractmethod
    def fit(X, y):
        pass

    @abstractmethod
    def predict(X):
        pass