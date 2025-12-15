import pandas as pd
import pandas.api.types
import numpy as np
from typing import Union
from lifelines.utils import concordance_index


def stratified_concordance_index(
    solution: pd.DataFrame,
    predictions: Union[pd.Series, np.ndarray],
    event_binary_col_name: str,
    duration_col_name: str,
    group_col_name: str,
) -> float:
    """
    Solution dataframe should contain all necessary columns
    """

    solution["predictions"] = predictions
    solution.reset_index(inplace=True)
    solution_group_dict = dict(solution.groupby([group_col_name]).groups)
    metric_list = []

    for race in solution_group_dict.keys():
        indices = sorted(solution_group_dict[race])
        merged_df_race = solution.iloc[indices]

        c_index_race = concordance_index(
            merged_df_race[duration_col_name],
            -merged_df_race["predictions"],
            merged_df_race[event_binary_col_name],
        )
        metric_list.append(c_index_race)
    return float(np.mean(metric_list) - np.sqrt(np.var(metric_list)))
