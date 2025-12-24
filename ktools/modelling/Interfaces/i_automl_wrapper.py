from functools import reduce
import os
import random
from typing import Any, List, Union
import torch
from abc import ABC, abstractmethod
from ktools.hyperopt.i_sklearn_kfold_object import (
    ISklearnKFoldObject,
)
from ktools.utils.data_science_pipeline_settings import DataSciencePipelineSettings
from ktools.preprocessing.basic_feature_transformers import *


class IAutomlWrapper(ABC):
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
        random_state: int = 42,
        save_predictions: bool = True,
        save_path: str = "",
    ) -> None:
        self._train_csv_path = train_csv_path
        self._test_csv_path = test_csv_path
        self._target_col_name = target_col_name
        self._kfold_object = kfold_object
        self._data_transforms = data_transforms
        self._random_state = random_state
        self._save_predictions = save_predictions
        self._save_path = save_path
        self._set_random_seeds()
        self._set_model_name_and_save_paths(model_name)
        self.train_df, self.test_df = self._data_setup()
        self.model = self._model_setup()

    def _set_random_seeds(self):
        np.random.seed(self._random_state)
        random.seed(self._random_state)
        torch.manual_seed(self._random_state)

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

    @abstractmethod
    def _set_model_name_and_save_paths(self, model_name):
        self._model_name = model_name
        self._oof_save_path = os.path.join(self._save_path, f"{model_name}_oof.csv")
        self._test_save_path = os.path.join(self._save_path, f"{model_name}_test.csv")

    @abstractmethod
    def _model_setup(self):
        pass

    @abstractmethod
    def fit(self):
        pass

    @abstractmethod
    def predict(self, df: Union[pd.DataFrame, None] = None):
        pass
