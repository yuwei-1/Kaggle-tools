import pandas as pd
import logging
from typing import List, Tuple
from sklearn.decomposition import PCA


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class PCAFeatureCreator:
    """
    Condenses n features into k features that capture
    over some variance threshold of the original n features.
    """

    def __init__(
        self, feature_names: List[str], variance_threshold: float = 0.9
    ) -> None:
        self._feature_names = feature_names
        self._variance_threshold = variance_threshold
        self.pca_model = PCA(n_components=None)

    def _condense_features(
        self, train_features: pd.DataFrame, test_features: pd.DataFrame
    ):
        pca_train = self.pca_model.fit_transform(train_features)
        pca_test = self.pca_model.transform(test_features)
        mask = (
            self.pca_model.explained_variance_ratio_.cumsum()
            <= self._variance_threshold
        )
        pca_train = pca_train[:, mask]
        pca_test = pca_test[:, mask]
        return pca_train, pca_test, mask.sum()

    def create(
        self, train_data: pd.DataFrame, test_data: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        train_features_to_reduce = train_data.loc[:, self._feature_names]
        test_features_to_reduce = test_data.loc[:, self._feature_names]
        pca_train, pca_test, num_components = self._condense_features(
            train_features_to_reduce, test_features_to_reduce
        )

        pca_train = pd.DataFrame(
            index=train_data.index,
            data=pca_train,
            columns=[f"PCA component {i}" for i in range(pca_train.shape[1])],
        )
        pca_test = pd.DataFrame(
            index=test_data.index,
            data=pca_test,
            columns=[f"PCA component {i}" for i in range(pca_train.shape[1])],
        )

        train_data = pd.concat([train_data, pca_train], axis=1)
        test_data = pd.concat([test_data, pca_test], axis=1)

        logger.info(f" {num_components} PCA components picked.")
        return train_data, test_data
