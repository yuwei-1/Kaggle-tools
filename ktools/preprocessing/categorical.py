import pandas as pd
from ktools.base.preprocessor import BasePreprocessor
from sklearn.preprocessing import OrdinalEncoder
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
        copy[self.config.categorical_col_names] = copy[
            self.config.categorical_col_names
        ].where(~mask, self.encode_missing_value)
        return copy
