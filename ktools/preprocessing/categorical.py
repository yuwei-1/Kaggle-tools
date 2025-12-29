from typing import Dict
import pandas as pd
from ktools.base.preprocessor import BasePreprocessor
from sklearn.preprocessing import OrdinalEncoder, TargetEncoder
from ktools.config.dataset import DatasetConfig


class CategoricalEncoder(BasePreprocessor):
    name = "categorical-encoder"

    """
    NOTE: sklearn bug where missing values are not "handled correctly".
    If NaN is not present in training data, in the validation set,
    they will be encoded as the unknown value instead of the encoded_missing_value.
    """

    def __init__(
        self,
        config: DatasetConfig,
        handle_unknown: str = "use_encoded_value",
        unknown_value: int = -2,
        encoded_missing_value: int = -1,
        **encoder_kwargs,
    ) -> None:
        super().__init__(config)
        self.encode_missing_value = encoded_missing_value
        self.encoder = OrdinalEncoder(
            handle_unknown=handle_unknown,
            unknown_value=unknown_value,
            encoded_missing_value=encoded_missing_value,
            **encoder_kwargs,
        )

    def fit(self, data: pd.DataFrame) -> "CategoricalEncoder":
        self.encoder.fit(data[self.config.categorical_col_names])
        self._fitted = True
        return self

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        copy = data.copy()
        mask = copy[self.config.categorical_col_names].isna()
        copy[self.config.categorical_col_names] = self.encoder.transform(
            copy[self.config.categorical_col_names]
        ).astype(int)
        copy[self.config.categorical_col_names] = (
            copy[self.config.categorical_col_names]
            .where(~mask, self.encode_missing_value)
            .astype("category")
        )
        return copy


class CategoricalFrequencyEncoder(BasePreprocessor):
    freq_suffix = "_frequency_encoding"
    _nan_encoding = "nan"

    def __init__(self, config: DatasetConfig, encode_missing_value: int = 0):
        super().__init__(config)
        self.train_freq_mappings: Dict[str, Dict[int]] = {}
        self.encode_missing_value = encode_missing_value

    def fit(self, data: pd.DataFrame) -> "CategoricalFrequencyEncoder":
        for column in self.config.categorical_col_names:
            new_col_name = column + self.freq_suffix
            if new_col_name in self.config.training_col_names:
                raise ValueError("Frequency encoded column already exists")

            freq_map = (
                data[column]
                .astype(str)
                .fillna(self._nan_encoding)
                .value_counts(normalize=True)
                .to_dict()
            )
            self.train_freq_mappings[column] = freq_map
        self._fitted = True
        return self

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        copy = data.copy()
        for column in self.config.categorical_col_names:
            new_col_name = column + self.freq_suffix
            freq_map = self.train_freq_mappings[column]
            copy[new_col_name] = (
                copy[column]
                .astype(str)
                .fillna(self._nan_encoding)
                .map(freq_map)
                .fillna(self.encode_missing_value)
                .astype(float)
            )
        return copy


class CategoricalTargetEncoder(BasePreprocessor):
    def __init__(
        self,
        config: DatasetConfig,
        random_state: int = 42,
        cv: int = 5,
        smooth: int = 15,
    ):
        super().__init__(config)
        self.target_encoder = TargetEncoder(
            random_state=random_state, cv=cv, smooth=smooth
        )

    def fit(self, data: pd.DataFrame) -> "CategoricalTargetEncoder":
        self.target_encoder.fit(
            data[self.config.categorical_col_names], data[self.config.target_col_name]
        )
        self._fitted = True
        return self

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        copy = data.copy()
        copy[self.config.categorical_col_names] = self.target_encoder.transform(
            copy[self.config.categorical_col_names]
        ).astype("float32")
        return copy
