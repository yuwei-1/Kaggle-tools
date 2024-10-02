from typing import Dict
import numpy as np
import pandas as pd
from copy import deepcopy
from sklearn.model_selection import StratifiedKFold
from ktools.modelling.Interfaces.i_sklearn_model import ISklearnModel
from ktools.hyperparameter_optimization.i_sklearn_kfold_object import ISklearnKFoldObject


class CrossValidateTestSklearnModel:

    def __init__(self,
                 sklearn_model_instance : ISklearnModel,
                 evaluation_metrics : Dict[str, callable],
                 kfold_object : ISklearnKFoldObject = None,
                 use_test_as_valid=True,) -> None:
        
        self.model = sklearn_model_instance
        self._evaluation_metrics = evaluation_metrics
        self._metric_names = list(evaluation_metrics.keys())
        self._kf = kfold_object
        self._use_test_as_valid = use_test_as_valid
        self._num_metrics = len(self._metric_names)
        self._num_splits = 1 if kfold_object is None else kfold_object.get_n_splits()
        self._model_list = []

    def _fit_then_predict(self, X, y, X_test, y_test):
        validation_set = None
        if self._use_test_as_valid:
            validation_set = [X_test, y_test]
        model = deepcopy(self.model).fit(X, y, validation_set=validation_set)
        y_pred = model.predict(X_test)
        return y_pred, model

    def evaluate(self,
                 X_train, y_train,
                 X_test, y_test,
                 stratified_set=None):

        cv_results = np.zeros((self._num_splits, self._num_metrics))
        cv_scores = None
        oof_predictions = np.zeros(X_train.shape[0])
        stratified_set = stratified_set if stratified_set is not None else y_train

        if self._kf is not None:
            for i, (train_index, val_index) in enumerate(self._kf.split(X_train, stratified_set)):
                X_train_fold, X_val_fold = X_train.iloc[train_index], X_train.iloc[val_index]
                y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]

                y_pred, model = self._fit_then_predict(X_train_fold,
                                                       y_train_fold,
                                                       X_val_fold,
                                                       y_val_fold)
                self._model_list += [model]
                oof_predictions[val_index] = y_pred

                for j, metric in enumerate(self._metric_names):
                    score = self._evaluation_metrics[metric](np.array(y_val_fold), np.array(y_pred))
                    cv_results[i][j] = score
            
            cv_scores = pd.DataFrame(columns=self._metric_names, data=cv_results)
            cv_scores.describe()


        y_pred, self.model = self._fit_then_predict(X_train,
                                                     y_train,
                                                     X_test,
                                                     y_test)
        test_scores = {}
        for j, metric in enumerate(self._metric_names):
            score = self._evaluation_metrics[metric](np.array(y_test), np.array(y_pred))
            cv_score = self._evaluation_metrics[metric](y_train.to_numpy().squeeze(), oof_predictions)
            test_scores[metric] = score
            print(f"Final Model {metric}: {score:.6f}")
            print(f"{self._num_splits}-fold cross validation {metric}: ", cv_score)
        
        return self._model_list, cv_scores, test_scores, oof_predictions, np.array(y_pred)