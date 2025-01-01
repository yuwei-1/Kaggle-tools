

from typing import Callable
from matplotlib import pyplot as plt
from numpy import ndarray
from pandas import DataFrame, Series
from sklearn.linear_model import Ridge
from ktools.modelling.Interfaces.i_ensemble_method import IEnsembleMethod


class RidgeRegressionBlendingEnsemble(IEnsembleMethod):
    
    def __init__(self, 
                 oof_dataframe: DataFrame, 
                 train_labels: DataFrame | Series | ndarray, 
                 metric: Callable,
                 post_transform : Callable = [lambda x : x],
                 alpha : float = 0.1,
                 plot : bool = True,
                 **ridge_kwargs) -> None:
        
        super().__init__(oof_dataframe, train_labels, metric)
        self._alpha = alpha
        self._plot = plot
        self._post_transform = post_transform
        self._rkw = ridge_kwargs
        self.ridge_blender = Ridge(alpha=self._alpha, **self._rkw)

    def fit_weights(self):
        self.ridge_blender.fit(self._oofs, self._labels)
        blended_oof_preds = self.ridge_blender.predict(self._oofs)
        metric_value = self._metric(self._post_transform(blended_oof_preds), self._labels)

        print("#"*100)
        print(str(self._metric), " score of ridge blender: ", metric_value)
        print("#"*100)

        if self._plot:
            plt.figure()
            plt.barh(self._oofs.columns, self.ridge_blender.coef_)
            plt.show()