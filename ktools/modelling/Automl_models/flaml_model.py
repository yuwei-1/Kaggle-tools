import os
from flaml import AutoML
from typing import Any, Callable, List, Union
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from ktools.fitting.cross_validation_executor import CrossValidationExecutor
from ktools.hyperparameter_optimization.i_sklearn_kfold_object import (
    ISklearnKFoldObject,
)
from ktools.modelling.Interfaces.i_automl_wrapper import IAutomlWrapper
from ktools.modelling.Interfaces.i_ktools_model import IKtoolsModel
from ktools.preprocessing.basic_feature_transformers import *


class FLAMLModel(IKtoolsModel):
    def __init__(
        self,
        time_budget: float = 60,
        metric: str = "mse",
        task: str = "regression",
        n_jobs: int = 1,
        estimator_list: List[str] = [
            "lgbm",
            "xgboost",
            "xgb_limitdepth",
            "catboost",
            "rf",
            "extra_tree",
        ],
        verbose: int = 0,
        seed: int = 42,
        **extra_params,
    ) -> None:
        self._random_state = seed
        self._task = task
        self._time_budget = time_budget
        self._model_parameters = {
            "time_budget": time_budget,
            "metric": metric,
            "task": task,
            "n_jobs": n_jobs,
            "estimator_list": estimator_list,
            "verbose": verbose,
            "seed": seed,
            **extra_params,
        }
        self.model = AutoML(**self._model_parameters)

    def fit(self, X, y, validation_set=None, val_size=0.05):
        if validation_set is None:
            X_train, X_valid, y_train, y_valid = train_test_split(
                X, y, test_size=val_size, random_state=self._random_state
            )
        else:
            X_train, y_train = X, y
            X_valid, y_valid = validation_set

        self.model.fit(
            X_train=X_train,
            y_train=y_train.values,
            X_val=X_valid,
            y_val=y_valid.values,
            time_budget=self._time_budget,
        )
        return self

    def predict(self, X):
        if self._task == "regression":
            y_pred = self.model.predict(X)
        elif self._task == "classification":
            y_pred = self.model.predict_proba(X)
        else:
            raise NotImplementedError
        return y_pred


class KToolsFLAMLWrapper(IAutomlWrapper):
    def __init__(
        self,
        train_csv_path: str,
        test_csv_path: str,
        target_col_name: str,
        kfold_object: ISklearnKFoldObject,
        data_transforms: List[Any] = [
            FillNullValues.transform,
            ConvertObjectToCategorical.transform,
        ],
        model_name: Union[str, None] = None,
        metric_callable: Callable = mean_squared_error,
        total_time_budget: float = 60,
        metric: str = "mse",
        task: str = "regression",
        n_jobs: int = 1,
        estimator_list: List[str] = [
            "lgbm",
            "xgboost",
            "xgb_limitdepth",
            "catboost",
            "rf",
            "extra_tree",
        ],
        verbose: int = 0,
        random_state: int = 42,
        save_predictions: bool = True,
        save_path: str = "",
        **extra_params,
    ) -> None:
        self._metric_callable = metric_callable
        self._total_time_budget = total_time_budget
        self._metric = metric
        self._task = task
        self._n_jobs = n_jobs
        self._estimator_list = estimator_list
        self._verbose = verbose
        self._random_state = random_state
        self._extra_params = extra_params
        self.num_folds = kfold_object.get_n_splits()

        super().__init__(
            train_csv_path,
            test_csv_path,
            target_col_name,
            kfold_object,
            data_transforms,
            model_name,
            random_state,
            save_predictions,
            save_path,
        )

    def _set_model_name_and_save_paths(self, model_name):
        self._model_name = model_name if model_name is not None else "FLAML"
        self._oof_save_path = os.path.join(
            self._save_path, f"{model_name}_flaml_oof.csv"
        )
        self._test_save_path = os.path.join(
            self._save_path, f"{model_name}_flaml_test.csv"
        )

    def _model_setup(self):
        return None

    def fit(self):
        model = FLAMLModel(
            time_budget=self._total_time_budget // self.num_folds,
            metric=self._metric,
            task=self._task,
            n_jobs=self._n_jobs,
            estimator_list=self._estimator_list,
            verbose=self._verbose,
            seed=self._random_state,
            **self._extra_params,
        )

        X, y = (
            self.train_df.drop(columns=self._target_col_name),
            self.train_df[[self._target_col_name]],
        )
        score_tuple, self.oofs, self.model_list = CrossValidationExecutor(
            model, self._metric_callable, self._kfold_object, verbose=self._verbose
        ).run(X, y)
        return self

    def predict(self, df: Union[pd.DataFrame, None] = None):
        if df is not None:
            test_pred = np.zeros(df.shape[0])
            for model in self.model_list:
                test_pred += model.predict(df) / self.num_folds
            all_y_preds = pd.DataFrame({f"{self._model_name}_flaml": test_pred})
            if self._save_predictions:
                all_y_preds.to_csv(self._test_save_path)
        else:
            all_y_preds = pd.DataFrame({f"{self._model_name}_flaml": self.oofs})
            if self._save_predictions:
                all_y_preds.to_csv(self._oof_save_path)

        return all_y_preds
