from functools import reduce
import os
from typing import Any, Dict, List, Union
from autogluon.tabular import TabularPredictor
from ktools.hyperopt.i_sklearn_kfold_object import (
    ISklearnKFoldObject,
)
from ktools.modelling.Interfaces.i_automl_wrapper import IAutomlWrapper
from ktools.modelling.model_transform_wrappers.survival_model_wrapper import (
    SupportedSurvivalTransformation,
)
from ktools.preprocessing.basic_feature_transformers import *


class KToolsAutogluonWrapper(IAutomlWrapper):
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
        eval_metric: str = "accuracy",
        problem_type: str = "binary",
        random_state: int = 42,
        included_model_types: List[str] = ["CAT", "GBM", "XGB"],
        fit_kwargs: Dict[str, Any] = {
            "verbosity": 2,
            "num_cpus": 4,
            "num_gpus": 2,
            "presets": "best_quality",
            "time_limit": 3600 * 11,
        },
        save_predictions: bool = True,
        save_path: str = "",
    ) -> None:
        self._included_model_types = included_model_types
        self._eval_metric = eval_metric
        self._problem_type = problem_type
        self._fit_kwargs = fit_kwargs

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
        self._model_name = (
            model_name
            if model_name is not None
            else "_".join(self._included_model_types)
        )
        self._oof_save_path = os.path.join(self._save_path, f"{model_name}_ag_oof.csv")
        self._test_save_path = os.path.join(
            self._save_path, f"{model_name}_ag_test.csv"
        )

    def _model_setup(self):
        kfold_col_name = "fold"

        X, y = (
            self.train_df.drop(columns=self._target_col_name),
            self.train_df[self._target_col_name],
        )
        split = self._kfold_object.split(X, y)
        for i, (_, val_index) in enumerate(split):
            self.train_df.loc[val_index, kfold_col_name] = i

        predictor = TabularPredictor(
            label=self._target_col_name,
            eval_metric=self._eval_metric,
            problem_type=self._problem_type,
            groups=kfold_col_name,
        )
        return predictor

    def load_model(self, ag_model_path: str):
        self.model = TabularPredictor.load(ag_model_path)
        return self

    def fit(self):
        self.model = self.model.fit(
            self.train_df,
            included_model_types=self._included_model_types,
            **self._fit_kwargs,
        )
        return self

    def predict(self, df: Union[pd.DataFrame, None] = None):
        if df is not None:
            all_y_preds = self.model.predict_multi(df)
            all_y_preds = pd.DataFrame.from_dict(all_y_preds)
            if self._save_predictions:
                all_y_preds.to_csv(self._test_save_path)
        else:
            all_y_preds = self.model.predict_multi()
            all_y_preds = pd.DataFrame.from_dict(all_y_preds)
            if self._save_predictions:
                all_y_preds.to_csv(self._oof_save_path)

        return all_y_preds


class KToolsSurvivalAutogluon(KToolsAutogluonWrapper):
    def __init__(
        self,
        *args,
        transform_string: str = "quantile",
        group_col_name: str = "race_group",
        times_col: str = "efs_time",
        event_col: str = "efs",
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._transform_string = transform_string
        self._group_col_name = group_col_name
        self._times_col = times_col
        self._event_col = event_col

    def _data_setup(self):
        settings = DataSciencePipelineSettings(
            self._train_csv_path,
            self._test_csv_path,
            self._target_col_name,
        )

        settings = reduce(lambda acc, func: func(acc), self._data_transforms, settings)
        train_df, test_df = settings.update()
        if not isinstance(self._target_col_name, list):
            self._target_col_name = [self._target_col_name]
        test_df.drop(columns=self._target_col_name, inplace=True)
        return train_df, test_df

    def _model_setup(self):
        kfold_col_name = "fold"
        self.transform = SupportedSurvivalTransformation[
            self._transform_string.upper()
        ].value

        X, y = (
            self.train_df.drop(columns=self._target_col_name),
            self.train_df[self._target_col_name],
        )
        groups = X[self._group_col_name]

        split = self._kfold_object.split(X, groups, groups=groups)
        for i, (train_index, val_index) in enumerate(split):
            self.train_df.loc[val_index, kfold_col_name] = i

        predictor = TabularPredictor(
            label=self._target_col_name,
            eval_metric=self._eval_metric,
            problem_type=self._problem_type,
            groups=kfold_col_name,
        )
        return predictor
