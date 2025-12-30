from typing import List, Optional, Tuple, Callable
import numpy as np
import pandas as pd
from copy import deepcopy
from ktools.config.dataset import DatasetConfig
from ktools.fitting.pipe import ModelPipeline
from ktools.hyperopt.i_sklearn_kfold_object import (
    ISklearnKFoldObject,
)


class CrossValidationExecutor:
    def __init__(
        self,
        config: DatasetConfig,
        model_pipeline: ModelPipeline,
        evaluation_metric: Callable,
        kfold_object: ISklearnKFoldObject,
    ) -> None:
        self.config = config
        self.model_pipeline = model_pipeline
        self._evaluation_metric = evaluation_metric
        self._splitter = kfold_object
        self._num_splits = kfold_object.get_n_splits()

    def run(
        self,
        train_data: pd.DataFrame,
        weights: Optional[np.ndarray] = None,
        val_data: Optional[pd.DataFrame] = None,
        test_data: Optional[pd.DataFrame] = None,
        groups=None,
        additional_data: Optional[pd.DataFrame] = None,
    ) -> Tuple[float, np.ndarray, List[ModelPipeline], np.ndarray]:
        train_oof_preds = np.empty(train_data.shape[0])
        test_oof_preds = np.zeros(test_data.shape[0])

        mean_score: int = 0
        pipelist: List[ModelPipeline] = []
        for train_index, val_index in self._splitter.split(
            train_data, train_data[self.config.target_col_name]
        ):
            train_fold = train_data.iloc[train_index]
            val_fold = train_data.iloc[val_index]

            pipe = deepcopy(self.model_pipeline)
            all_training_data = (
                pd.concat([train_fold, additional_data])
                if additional_data is not None
                else train_fold
            )
            validation_data = val_fold if val_data is None else val_data
            pipe.fit(
                all_training_data, validation_data=validation_data, weights=weights
            )
            pipelist.append(pipe)

            y_pred = pipe.predict(val_fold.drop(columns=[self.config.target_col_name]))
            test_pred = pipe.predict(test_data)

            score = self._evaluation_metric(
                val_fold[self.config.target_col_name], y_pred
            )
            train_oof_preds[val_index] = y_pred
            test_oof_preds += test_pred / self._num_splits

            mean_score += score / self._num_splits

        return mean_score, train_oof_preds, pipelist, test_oof_preds
