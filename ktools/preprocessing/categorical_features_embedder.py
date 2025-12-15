import pandas as pd
from typing import List


class SortMainCategories:
    """
    Convert non-numerical categorical data to ordinal feature.
        df: training dataframe
        col_names_to_categorize: column names that need to change data from object to numerical
        cutoff: the minimum frequency that is considered to be a category
        fill_with_most_likely: fill with the mode/mean of column
    """

    special_value = -1

    def __init__(
        self,
        df: pd.DataFrame,
        col_names_to_categorize: List,
        cutoff: int,
        fill_with_most_likely: bool = False,
        verbose: bool = False,
    ) -> None:
        self.df = df
        self.col_names_to_categorize = set(col_names_to_categorize)
        self.cutoff = cutoff
        self.fill_with_most_likely = fill_with_most_likely
        self.verbose = verbose
        self.accepted_values = self._find_accepted_values()

    def _find_accepted_values(self):
        accepted_values = {}
        for col in self.df.columns:
            if col in self.col_names_to_categorize:
                k = (self.df[col].value_counts().values > self.cutoff).sum()
                if self.verbose:
                    print(col, self.df[col].value_counts().head(k).index.tolist())
                    print(self.df[col].value_counts().head(k).values)
                allowed_list = self.df[col].value_counts().head(k).index.tolist()
                accepted_values[col] = {allowed_list[i]: i for i in range(k)}
        return accepted_values

    def sort(self, dataframe_to_sort):
        return self.encode_columns(
            dataframe_to_sort, self.accepted_values, self.special_value
        )

    def encode_columns(self, df, accepted_values, special_value):
        encoded_df = df.copy()
        for col, mapping in accepted_values.items():
            if col in encoded_df.columns:
                encoded_df[col] = df[col].map(mapping)

                if self.fill_with_most_likely:
                    if encoded_df[col].dtype == "object":
                        mode_value = encoded_df[col].mode()[0]
                        encoded_df[col] = encoded_df[col].fillna(mode_value).astype(int)
                    else:
                        mean_value = encoded_df[col].mean()
                        encoded_df[col] = encoded_df[col].fillna(mean_value)
                else:
                    encoded_df[col] = encoded_df[col].fillna(special_value).astype(int)
        return encoded_df
