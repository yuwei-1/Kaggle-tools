import numpy as np
from ktools.modelling.Interfaces.i_model_wrapper import IModelWrapper
from ktools.modelling.Interfaces.i_ktools_model import IKtoolsModel


class LogModelWrapper(IModelWrapper):

    def __init__(self, model: IKtoolsModel) -> None:
        super().__init__(model)

    def fit(self, X, y, *args, validation_set = None, **kwargs):
        X_valid, y_valid = validation_set
        
        y = np.log1p(y)
        self.model.fit(X, y, *args, **kwargs)

    def predict(self, X, *args, **kwargs):
        y_pred = self.model.predict(X, *args, **kwargs)
        return np.expm1(y_pred)