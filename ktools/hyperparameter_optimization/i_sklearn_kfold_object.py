from abc import ABC, abstractmethod


class ISklearnKFoldObject(ABC):
    random_state : int = 0

    @abstractmethod
    def split(X, y):
        pass

    @abstractmethod
    def get_n_splits(self):
        pass