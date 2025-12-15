import itertools
from typing import List, Tuple
import pandas as pd
from ktools.feature_engineering.interfaces.i_feature_creator import IFeatureCreator


class FrequencyEncodingCreator(IFeatureCreator):
    @staticmethod
    def create(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        categorical_col_names = df.select_dtypes(
            ["object", "category"]
        ).columns.tolist()
        count_cols = [col + "_count" for col in categorical_col_names]
        for i in range(len(count_cols)):
            freq_dict = df[categorical_col_names[i]].value_counts().to_dict()
            df[count_cols[i]] = df[categorical_col_names[i]].map(freq_dict)
        return df, count_cols


class CategoricalCombinationsCreator(IFeatureCreator):
    @staticmethod
    def create(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        categorical_col_names = df.select_dtypes(
            ["object", "category"]
        ).columns.tolist()
        combinations = list(itertools.combinations(categorical_col_names, 2))
        new_col_names = []
        for feature_1, feature_2 in combinations:
            new_feature_name = feature_1 + "_" + feature_2
            new_col_names += [new_feature_name]
            df[new_feature_name] = (
                df[feature_1].astype("str") + "_" + df[feature_2].astype("str")
            )
        return df, new_col_names


class NumericalToCategoricalCreator(IFeatureCreator):
    def __init__(self, target_col_name: str) -> None:
        self._target_col_name = target_col_name

    def create(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        columns = [x for x in df.columns if x != self._target_col_name]
        added_feats = []
        for col in columns:
            if df[col].dtype != "category":
                new_col = "cat_" + col
                added_feats += [new_col]
                df[new_col] = df[col].astype(str).astype("category")
        return df, added_feats
