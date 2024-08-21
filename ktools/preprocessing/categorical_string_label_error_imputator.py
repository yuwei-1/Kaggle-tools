from typing import List
import pandas as pd
import numpy as np
import re


class CategoricalLabelErrorImputator:

    def __init__(self,
                 verbose=False
                 ) -> None:
        self._verbose = verbose

    def impute(self,
               dataframe : pd.DataFrame,
               col_names_to_impute : List[str],
               threshold : int = 10):
        
        for col_name in col_names_to_impute:
            relabeller = {}
            dataframe[col_name] = dataframe[col_name].str.lower()
            val_counts = dataframe[col_name].value_counts(sort=True)
            unique_values = val_counts.index
            occurrences = val_counts.values
            above_threshold = occurrences > threshold
            valid_classes = unique_values[above_threshold]

            for val in unique_values[~above_threshold]:
                try:
                    extracted_components = np.array(re.split(r'\W+', val))
                    found_idcs = np.where(np.isin(valid_classes, extracted_components))[0]
                    if found_idcs.size > 0:
                        relabeller[val] = valid_classes[found_idcs[0]]
                except:
                    continue
            if self._verbose:
                print("(original : new category)", relabeller)
            dataframe[col_name] = dataframe[col_name].map(lambda x: relabeller.get(x, x))

        return dataframe