from abc import ABC, abstractmethod


class ISklearnKFoldObject(ABC):

    @abstractmethod
    def split(X, y):
        pass