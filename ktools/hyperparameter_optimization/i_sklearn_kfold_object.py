from abc import ABC, abstractmethod


class ISklearnKFoldObject(ABC):

    @abstractmethod
    def split(X, y):
        pass

    @abstractmethod
    def get_n_splits(self):
        pass