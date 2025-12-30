import numpy as np
import pandas as pd
from ktools.base.model import BaseKtoolsModel
from typing import Optional, Union
from ktools.config.dataset import DatasetConfig
from ktools.preprocessing.pipe import PreprocessingPipeline


class ModelPipeline:
    def __init__(
        self,
        model: BaseKtoolsModel,
        config: DatasetConfig,
        preprocessor: PreprocessingPipeline = PreprocessingPipeline([]),
    ) -> None:
        self.model = model
        self.config = config
        self.preprocessor = preprocessor

    def fit(
        self,
        train_data: pd.DataFrame,
        validation_data: Optional[pd.DataFrame] = None,
        weights: Optional[Union[pd.Series, np.ndarray]] = None,
        val_weights: Optional[Union[pd.Series, np.ndarray]] = None,
    ) -> "ModelPipeline":
        train_data = self.preprocessor.train_pipe(train_data)
        X_train = train_data.drop(columns=[self.config.target_col_name])
        y_train = train_data[self.config.target_col_name]

        if validation_data is not None:
            validation_data = self.preprocessor.inference_pipe(validation_data)
            X_valid = validation_data.drop(columns=[self.config.target_col_name])
            y_valid = validation_data[self.config.target_col_name]
            validation_data = (X_valid, y_valid)

        self.model.fit(
            X=X_train,
            y=y_train,
            validation_set=validation_data,
            weights=weights,
            val_weights=val_weights,
        )
        return self

    def predict(self, data: pd.DataFrame) -> np.ndarray:
        data = self.preprocessor.inference_pipe(data)
        X_test = data  # [self.config.training_col_names]
        return self.model.predict(X_test)
