from dataclasses import dataclass
from typing import *
import pandas as pd


@dataclass
class DataSciencePipelineSettings(object):
    train_csv_path : str
    test_csv_path : str
    target_col_name : List[str]
    original_csv_path : str = None

    train_data : Union[None, pd.DataFrame] = None
    test_data : Union[None, pd.DataFrame] = None
    original_data : Union[None, pd.DataFrame] = None

    original_csv_processing : callable = lambda x : x
    sample_submission_path : str = None
    training_col_names : List[str] = None
    categorical_col_names : List[str] = None
    training_data_percentage : float = 0.8
    category_occurrence_threshold : int = 300
    logged : bool = False

    def __post_init__(self):
        self.train_df, self.test_df = self._gather_data()
        self.training_col_names, self.categorical_col_names, self.numerical_col_names = self._get_column_info()
        self.combined_df = self._combine_datasets()

    def _gather_data(self):
        train_df = self._smart_drop_index(pd.read_csv(self.train_csv_path) if self.train_data is None else self.train_data)
        test_df = self._smart_drop_index(pd.read_csv(self.test_csv_path) if self.test_data is None else self.test_data)

        if self.original_csv_path or self.original_data:
            train_df = train_df.assign(source=0)
            test_df = test_df.assign(source=0)
            original_df = self._smart_drop_index(pd.read_csv(self.original_csv_path) if self.original_data is None else self.original_data).assign(source=1)
            original_df = self.original_csv_processing(original_df)

            pd.testing.assert_index_equal(train_df.columns.sort_values(), original_df.columns.sort_values(), check_exact=True)
            pd.testing.assert_series_equal(train_df.dtypes.sort_index(), original_df.dtypes.sort_index(), check_exact=True)
            train_df = pd.concat([train_df, original_df], axis=0).reset_index(drop=True)

        return train_df, test_df
    
    def _get_column_info(self):
        cat_col_names = [col_name for col_name in self.train_df.columns if self.train_df[col_name].dtype == 'object']
        training_features = list(self.train_df.drop(columns=self.target_col_name).columns)
        cat_col_names = cat_col_names if self.categorical_col_names is None else self.categorical_col_names
        num_col_names = [f for f in training_features if f not in cat_col_names]
        return training_features, cat_col_names, num_col_names
    
    def _combine_datasets(self):
        combined_df = pd.concat([self.train_df, self.test_df], keys=['train', 'test'])
        return combined_df
    
    def update(self):
        self.train_df = self.combined_df.loc['train'].copy()
        self.test_df = self.combined_df.loc['test'].copy()
        return self.train_df, self.test_df
    
    def get_data(self):
        self.update()
        X_test, y_test = self.test_df.drop(columns=self.target_col_name), self.test_df[self.target_col_name]
        X, y = self.train_df.drop(columns=self.target_col_name), self.train_df[self.target_col_name]
        return X, y, X_test, y_test

    @staticmethod
    def _smart_drop_index(df):
        try:
            differences = df.iloc[:, 0].diff().dropna()
            if differences.nunique() == 1:
                df = df.drop(columns=df.columns[0])
        except:
            pass
        return df
    
    @property
    def target_col(self):
        """target column name property."""
        return self.target_col_name

    @target_col.setter
    def target_col(self, value):
        self.target_col_name = value