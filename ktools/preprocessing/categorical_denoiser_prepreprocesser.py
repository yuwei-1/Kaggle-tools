from ktools.preprocessing.basic_preprocessor import BasicPreprocessor
from ktools.preprocessing.categorical_features_embedder import SortMainCategories


class CategoricalDenoiserPreprocessor(BasicPreprocessor):
    def _process_categorical_columns(self, train_dataframe, test_dataframe):
        sortmaincategories = SortMainCategories(
            train_dataframe,
            self._dss.categorical_col_names + [self._dss.target_col_name],
            self._dss.category_occurrence_threshold,
            self._verbose,
        )

        train_dataframe, test_dataframe = (
            sortmaincategories.sort(train_dataframe),
            sortmaincategories.sort(test_dataframe),
        )
        return train_dataframe, test_dataframe
