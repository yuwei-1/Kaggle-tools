import pandas as pd
from typing import Tuple
from ktools.preprocessing.i_preprocessing_utility import IPreprocessingUtility
from ktools.preprocessing.categorical_string_label_error_imputator import (
    CategoricalLabelErrorImputator,
)
from ktools.utils.data_science_pipeline_settings import DataSciencePipelineSettings


class BasicPreprocessor(IPreprocessingUtility):
    def __init__(
        self,
        data_science_settings: DataSciencePipelineSettings,
        return_categorical: bool = False,
        verbose: bool = False,
    ) -> None:
        super().__init__(data_science_settings, return_categorical, verbose)

    def process(self) -> Tuple[pd.DataFrame, pd.DataFrame, DataSciencePipelineSettings]:
        kaggle_train_data = pd.read_csv(self._dss.train_csv_path, index_col=0)
        kaggle_test_data = pd.read_csv(self._dss.test_csv_path, index_col=0)

        test_data_col_names = kaggle_test_data.columns
        train_data_col_names = kaggle_train_data.drop(
            columns=self._dss.target_col_name
        ).columns

        if self._verbose:
            non_matching_col_names = set(test_data_col_names) ^ set(
                train_data_col_names
            )
            print(
                f"The following columns are not consistent across data sets: {non_matching_col_names}"
            )

        self._dss.training_col_names = list(
            set(test_data_col_names).intersection(set(train_data_col_names))
        )
        self._dss.categorical_col_names = [
            col_name
            for col_name in self._dss.training_col_names
            if kaggle_train_data[col_name].dtype == "object"
        ]

        kaggle_train_data = CategoricalLabelErrorImputator(
            verbose=self._verbose
        ).impute(
            kaggle_train_data.copy(),
            self._dss.categorical_col_names,
            self._dss.category_occurrence_threshold,
        )
        kaggle_test_data = CategoricalLabelErrorImputator(verbose=self._verbose).impute(
            kaggle_test_data.copy(),
            self._dss.categorical_col_names,
            self._dss.category_occurrence_threshold,
        )
        kaggle_train_data, kaggle_test_data = self._process_categorical_columns(
            kaggle_train_data, kaggle_test_data
        )
        if self._return_categorical:
            kaggle_train_data, kaggle_test_data = (
                self._convert_to_categorical(kaggle_train_data),
                self._convert_to_categorical(kaggle_test_data),
            )
        return kaggle_train_data, kaggle_test_data, self._dss

    def _process_categorical_columns(self, train_dataframe, test_dataframe):
        return train_dataframe, test_dataframe

    def _convert_to_categorical(self, dataframe):
        dataframe[self._dss.categorical_col_names] = dataframe[
            self._dss.categorical_col_names
        ].astype("category")
        return dataframe
