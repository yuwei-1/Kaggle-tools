from functools import reduce
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd
from copy import deepcopy
from ktools.modelling.Interfaces.i_sklearn_model import ISklearnModel
from ktools.hyperparameter_optimization.i_sklearn_kfold_object import ISklearnKFoldObject



class CrossValidationExecutor:

    def __init__(self,
                 sklearn_model_instance : ISklearnModel,
                 evaluation_metric : callable,
                 kfold_object : ISklearnKFoldObject,
                 use_test_as_valid=True,
                 num_classes=None,
                 verbose=1) -> None:
        
        self.model = sklearn_model_instance
        self._evaluation_metric = evaluation_metric
        self._kf = kfold_object
        self._num_splits = kfold_object.get_n_splits()
        self._use_test_as_valid = use_test_as_valid
        self._num_classes = num_classes
        self._verbose = verbose

    def run(self, X, y, additional_data=None, local_transform_list=[lambda x : x], output_transform_list=[lambda x : x]) -> Tuple[Tuple[float], np.ndarray, List[Any]]:
        if additional_data is not None:
            X_add, y_add = additional_data
            pd.testing.assert_index_equal(X.columns, X_add.columns, check_exact=True)
            pd.testing.assert_series_equal(X.dtypes, X_add.dtypes, check_exact=True)
            pd.testing.assert_index_equal(y.columns, y_add.columns, check_exact=True)
            pd.testing.assert_series_equal(y.dtypes, y_add.dtypes, check_exact=True)

        cv_results = []
        model_list = []
        oof_predictions = np.zeros(y.shape[0]) if self._num_classes is None else np.zeros((y.shape[0], self._num_classes))

        for i, (train_index, val_index) in enumerate(self._kf.split(X, y)):
            
            X_train, X_test = X.iloc[train_index], X.iloc[val_index]
            y_train, y_test = y.iloc[train_index], y.iloc[val_index]

            if additional_data is not None:
                X_train = pd.concat([X_train, X_add], axis=0)
                y_train = pd.concat([y_train, y_add], axis=0)

            X_train, y_train = reduce(lambda acc, func: func(acc), local_transform_list, (X_train, y_train))
            validation_set = None
            if self._use_test_as_valid:
                validation_set = [X_test, y_test]

            model = deepcopy(self.model).fit(X_train, y_train, validation_set=validation_set)
            model_list += [model]
            y_pred = reduce(lambda acc, func: func(acc), output_transform_list, model.predict(X_test))
            
            cv_results += [self._evaluation_metric(y_test, y_pred)]
            oof_predictions[val_index] = y_pred

            if self._verbose > 1:
                print(f"The CV results of the current fold is {cv_results[-1]}")

        oof_score = self._evaluation_metric(y, oof_predictions)
        mean_cv_score = np.mean(cv_results)
        score_tuple = (oof_score, mean_cv_score)

        if self._verbose > 0:
            print("#"*100)
            print("OOF prediction score : ", oof_score)
            print(f"Mean {self._num_splits}-cv results : {mean_cv_score} +- {np.std(cv_results)}")
            print("#"*100)

        return score_tuple, oof_predictions, model_list