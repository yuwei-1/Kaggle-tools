from typing import *
import numpy as np
import catboost as cat
from catboost import Pool
import pandas as pd
from ktools.modelling.base_classes.base_ktools_model import BaseKtoolsModel
from ktools.modelling.base_classes.joblib_saver_mixin import JoblibSaverMixin


class CatBoostModel(BaseKtoolsModel, JoblibSaverMixin):

    def __init__(self,
                 num_boost_round : int = 100,
                 early_stopping_rounds : Union[int, None] = 20,
                 random_state : int = 129,
                 predict_type : Union[str, None] = None,
                 verbose : bool = False,
                 allow_writing_files : bool = False,
                 **catboost_params) -> None:
        
        super().__init__(random_state, early_stopping_rounds)
        self._num_boost_round = num_boost_round
        self._predict_type = predict_type
        self._verbose = verbose
        self._allow_writing_files = allow_writing_files

        self._catboost_params = {
            "random_seed" : random_state,
            "verbose" : verbose,
            "allow_writing_files" : allow_writing_files,
            **catboost_params
        }
        self.is_fitted_ = False

    def _infer_task(self, y):
        """
        Will infer binary classification if the target only has two unique values.
        Assumes regression otherwise.
        """
        uniques = np.unique(y).shape[0]
        if uniques == 2:
            self._predict_type = "prob"
        else:
            self._predict_type = "else"

    def fit(self, X: np.ndarray, y: np.ndarray, validation_set: Union[Tuple[np.ndarray, np.ndarray], None] = None, 
            val_size: float = 0.05, weights: Union[np.ndarray, None] = None) -> "CatBoostModel":
        
        if self._predict_type is None: self._infer_task(y)
        self.cat_col_names = [col for col in X.columns if X[col].dtype == 'category'] if isinstance(X, pd.DataFrame) else []
        X_train, X_valid, y_train, y_valid, weights = self.create_validation(X, y, validation_set, weights, val_size)
        
        train_params = {'eval_set' : None}
        train_pool = Pool(data=X_train, label=y_train, cat_features=self.cat_col_names, weight=weights)
        if self._early_stop:
            train_params['eval_set'] = Pool(data=X_valid, label=y_valid, cat_features=self.cat_col_names)
            train_params['early_stopping_rounds'] = self._early_stopping_rounds

        train_params = {
            "params" : self._catboost_params,
            "dtrain" : train_pool,
            "num_boost_round" : self._num_boost_round,
            **train_params
        }
        self.model = cat.train(
            **train_params
        )
        self.is_fitted_ = True
        return self

    def predict(self, X : np.ndarray) -> np.ndarray:
        test_pool = Pool(data=X, cat_features=self.cat_col_names)
        if self._predict_type == "prob":
            y_pred = self.model.predict(test_pool, prediction_type='Probability')[:, 1]
        elif self._predict_type == "class":
            y_pred = self.model.predict(test_pool, prediction_type='Class')
        else:
            y_pred = self.model.predict(test_pool)
        return y_pred

    @property
    def num_fitted_models(self):
        return self.model.tree_count_