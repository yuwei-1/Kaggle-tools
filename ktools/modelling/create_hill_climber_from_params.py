import numpy as np
from typing import Any, Dict, List
import pandas as pd
from ktools.fitting.cross_validation_executor import CrossValidationExecutor
from ktools.modelling.hill_climber_object import HillClimber


class CreateHillClimber:
    def __init__(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        model_list: List[Any],
        model_names: List[str],
        model_features: Dict[str, List[str]],
        eval_metric: callable,
        target_col_name: str,
        kfold,
        objective="minimize",
        negative_weights=False,
        plot_hill=False,
    ) -> None:
        self._train_df = train_df
        self._test_df = test_df
        self._model_list = model_list
        self._model_names = model_names
        self._model_features = model_features
        self._eval_metric = eval_metric
        self._target_col_name = target_col_name
        self._kfold = kfold
        self._objective = objective
        self._negative_weights = negative_weights
        self._plot_hill = plot_hill

    def fit(self):
        X, y = (
            self._train_df.drop(columns=self._target_col_name),
            self._train_df[self._target_col_name],
        )
        X_test, y_test = (
            self._test_df.drop(columns=self._target_col_name),
            self._test_df[self._target_col_name],
        )
        test_oof_list = []
        train_oof_list = []

        test_length = len(X_test)
        for model_name, model in zip(self._model_names, self._model_list):
            model_features = self._model_features[model_name]

            _X_train = X if model_features is None else X[model_features]
            _, train_oof, model_list = CrossValidationExecutor(
                model,
                self._eval_metric,
                self._kfold,
            ).run(_X_train, y)
            _X_test = X_test if model_features is None else X_test[model_features]
            test_oofs = np.zeros(test_length)
            for model in model_list:
                test_oofs += model.predict(_X_test) / len(model_list)

            test_oof_list += [(model_name, test_oofs)]
            train_oof_list += [(model_name, train_oof)]

        self.train_oof = train_oof = pd.concat(
            [pd.Series(arr, index=X.index, name=name) for name, arr in train_oof_list],
            axis=1,
        )
        self.test_preds = test_preds = pd.concat(
            [
                pd.Series(arr, index=X_test.index, name=name)
                for name, arr in test_oof_list
            ],
            axis=1,
        )

        hill_climber = HillClimber(
            self._train_df,
            train_oof,
            test_preds,
            self._target_col_name,
            self._eval_metric,
            self._objective,
            negative_weights=self._negative_weights,
            plot_hill=self._plot_hill,
        )
        return hill_climber
