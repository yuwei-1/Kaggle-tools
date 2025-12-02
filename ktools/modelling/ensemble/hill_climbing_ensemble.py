from typing import Callable
from hillclimbers import climb_hill
from functools import partial
import pandas as pd
from ktools.modelling.Interfaces.i_ensemble_method import IEnsembleMethod


class HillClimbingBlendingEnsemble(IEnsembleMethod):

    def __init__(self,
                 oof_dataframe : pd.DataFrame,
                 train_dataframe : pd.DataFrame,
                 test_dataframe : pd.DataFrame,
                 target_col_name : str,
                 metric : Callable,
                 objective : str = "minimize",
                 negative_weights : bool = True,
                 plot_hill : bool = True,
                 return_oof_preds : bool = False) -> None:
        
        super().__init__(oof_dataframe, train_dataframe[target_col_name], metric)

        self._oof_df = oof_dataframe
        self._train_df = train_dataframe
        self._test_df = test_dataframe
        self._target_col_name = target_col_name
        self._objective = objective
        self._negative_weights = negative_weights
        self._plot_hill = plot_hill
        self._return_oof_preds = return_oof_preds


    def fit_weights(self):

        best_test_pred, best_oof_pred = climb_hill(

            train=self._train_df,
            oof_pred_df=self._train_df[self._target_col_name],
            test_pred_df= self._test_df,
            target=self._target_col_name,
            eval_metric=partial(self._metric),
            objective=self._objective,
            negative_weights=self._negative_weights,
            plot_hill=self._plot_hill,
            return_oof_preds=self._return_oof_preds
        )

        return best_test_pred, best_oof_pred