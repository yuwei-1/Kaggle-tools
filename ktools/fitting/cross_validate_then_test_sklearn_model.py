from typing import Dict
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from ktools.fitting.interfaces.i_sklearn_model import ISklearnModel


class CrossValidateTestSklearnModel:

    def __init__(self,
                 sklearn_model_instance : ISklearnModel,
                 evaluation_metrics : Dict[str, callable],
                 kfold_object = None,
                 num_splits : int = 5) -> None:
        self.model = sklearn_model_instance
        self._evaluation_metrics = evaluation_metrics
        self._metric_names = list(evaluation_metrics.keys())
        self._kf = kfold_object
        self._num_metrics = len(self._metric_names)
        self._num_splits = num_splits

    def _fit_then_predict(self, X, y, X_test):
        self.model.fit(X, y)
        y_pred = self.model.predict(X_test)
        return y_pred

    def evaluate(self,
                 X_train, y_train,
                 X_test, y_test):

        cv_results = np.zeros((self._num_splits, self._num_metrics))
        cv_scores = None

        if self._kf is not None:
            for i, (train_index, val_index) in enumerate(self._kf.split(X_train, y_train)):
                X_train_fold, X_val_fold = X_train.iloc[train_index], X_train.iloc[val_index]
                y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]

                y_pred = self._fit_then_predict(X_train_fold,
                                                y_train_fold,
                                                X_val_fold)

                for j, metric in enumerate(self._metric_names):
                    score = self._evaluation_metrics[metric](np.array(y_val_fold), np.array(y_pred))
                    cv_results[i][j] = score
            
            cv_scores = pd.DataFrame(columns=self._metric_names, data=cv_results)
            cv_scores.describe()

        y_pred = self._fit_then_predict(X_train,
                                        y_train,
                                        X_test)
        test_scores = {}
        for j, metric in enumerate(self._metric_names):
            score = self._evaluation_metrics[metric](np.array(y_test), np.array(y_pred))
            test_scores[metric] = score
            print(f"Final Model {metric}: {score:.6f}")
            print(f"{self._num_splits}-fold cross validation {metric}: ", cv_results[:,j].mean())
        
        return self.model, cv_scores, test_scores