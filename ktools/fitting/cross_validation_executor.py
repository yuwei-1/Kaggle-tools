from functools import reduce
from typing import Any, Dict, List, Tuple, Callable, Union
import numpy as np
import pandas as pd
from copy import deepcopy
from ktools.modelling.Interfaces.i_ktools_model import IKtoolsModel
from ktools.hyperparameter_optimization.i_sklearn_kfold_object import ISklearnKFoldObject



class CrossValidationExecutor:

    def __init__(self,
                 sklearn_model_instance : IKtoolsModel,
                 evaluation_metric : Callable,
                 kfold_object : ISklearnKFoldObject,
                 training_features : Union[List[str], None] = None,
                 use_test_as_valid = True,
                 refit_on_all_training_for_test : bool = False,
                 num_classes = None,
                 verbose=1) -> None:
        
        self.model = sklearn_model_instance
        self._evaluation_metric = evaluation_metric
        self._kf = kfold_object
        self._num_splits = kfold_object.get_n_splits()
        self._training_features = training_features
        self._use_test_as_valid = use_test_as_valid
        self._refit_on_all_training_for_test = refit_on_all_training_for_test
        self._num_classes = num_classes
        self._verbose = verbose

    def run(self, X : pd.DataFrame, y : Union[pd.DataFrame, pd.Series], weights=None, test_data=None, groups = None, additional_data=None, local_transform_list=[lambda x : x], output_transform_list=[lambda x : x[-1]]) -> Tuple[Tuple[float], np.ndarray, List[Any]]:

        training_features = X.columns.tolist() if self._training_features is None else self._training_features
        X.reset_index(drop=True, inplace=True)
        y.reset_index(drop=True, inplace=True)

        if additional_data is not None:
            X_add, y_add = additional_data
            pd.testing.assert_index_equal(X.columns, X_add.columns, check_exact=True)
            pd.testing.assert_series_equal(X.dtypes, X_add.dtypes, check_exact=True)
            pd.testing.assert_index_equal(y.columns, y_add.columns, check_exact=True)
            pd.testing.assert_series_equal(y.dtypes, y_add.dtypes, check_exact=True)

        cv_results = []
        model_list = []
        oof_predictions = None
        metric_predictions = None
        test_predictions = None

        groups = y if groups is None else groups
        weights = np.ones(y.shape[0]) if weights is None else weights

        for i, (train_index, val_index) in enumerate(self._kf.split(X, groups, groups=groups)):
            
            X_full_test = X.loc[val_index, :]
            X_train, X_test = X.loc[train_index, training_features], X.loc[val_index, training_features]
            y_train, y_test = y.loc[train_index], y.loc[val_index]
            train_weights = weights[train_index]

            if additional_data is not None:
                X_train = pd.concat([X_train, X_add], axis=0)
                y_train = pd.concat([y_train, y_add], axis=0)

            X_train, y_train = reduce(lambda acc, func: func(acc), local_transform_list, (X_train, y_train))
            validation_set = None
            if self._use_test_as_valid:
                validation_set = [X_test, y_test]

            model = deepcopy(self.model).fit(X_train, y_train, validation_set=validation_set, weights=train_weights)
            model_list += [model]
            y_pred = model.predict(X_test)
            y_pred_processed = reduce(lambda acc, func: func(acc), output_transform_list, (X_full_test.copy(), y_pred))
            
            cv_results += [self._evaluation_metric(y_test, deepcopy(y_pred_processed))]

            if oof_predictions is None:
                oof_shape = (y.shape[0],) if len(y_pred.shape) == 1 else (y.shape[0], y_pred.shape[-1])
                oof_predictions = np.zeros(oof_shape)
            if metric_predictions is None:
                y_hat_shape = (y.shape[0],) if len(y_pred_processed.shape) == 1 else (y.shape[0], y_pred_processed.shape[-1])
                metric_predictions = np.zeros(y_hat_shape)

            if test_data is not None and not self._refit_on_all_training_for_test:
                test_preds = model.predict(test_data)
                if test_predictions is None: 
                    test_predictions = test_preds/self._num_splits
                else:
                    test_predictions += test_preds/self._num_splits

            oof_predictions[val_index] = y_pred
            metric_predictions[val_index] = y_pred_processed

            if self._verbose > 1:
                print(f"The CV results of the current fold is {cv_results[-1]}")

        if self._refit_on_all_training_for_test:
            model = deepcopy(self.model).fit(X_train, y_train, weights=weights)
            test_predictions = model.predict(test_data)

        oof_score = self._evaluation_metric(y, metric_predictions)
        mean_cv_score = np.mean(cv_results)
        score_tuple = (oof_score, mean_cv_score)

        if self._verbose > 0:
            print("#"*100)
            print("OOF prediction score : ", oof_score)
            print(f"Mean {self._num_splits}-cv results : {mean_cv_score} +- {np.std(cv_results)}")
            print("#"*100)

        return score_tuple, oof_predictions, model_list, test_predictions