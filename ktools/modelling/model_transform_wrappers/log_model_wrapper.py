import numpy as np
from ktools.modelling.Interfaces.i_model_wrapper import IModelWrapper
from ktools.modelling.Interfaces.i_sklearn_model import ISklearnModel


class LogModelWrapper(IModelWrapper):

    def __init__(self, model: ISklearnModel) -> None:
        super().__init__(model)

    def fit(self, X, y, *args, **kwargs):
        y = np.log1p(y)
        self.model.fit(X, y)

    def predict(self, X, *args, **kwargs):
        y_pred = self.model.predict(X)
        return np.expm1(y_pred)