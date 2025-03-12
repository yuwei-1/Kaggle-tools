import pandas as pd
from sklearn.preprocessing import MinMaxScaler, TargetEncoder
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from ktools.modelling.Interfaces.i_ktools_model import IKtoolsModel


class KNNModel(IKtoolsModel):

    def __init__(self, 
                 categorical_features, 
                 n_neighbors=5, 
                 weights="uniform", 
                 smooth="auto", 
                 target_type="continuous",
                 min_max_scaling=False,
                 random_state=129) -> None:
        self._n_neighbors = n_neighbors
        self._weights = weights
        self._target_enc = TargetEncoder(target_type=target_type, 
                                         smooth=smooth,
                                         random_state=random_state)
        self._categorical_features = categorical_features
        self._min_max_scaling = min_max_scaling
        self._minmax_scaler = MinMaxScaler()

        if target_type == "continuous":
            self.model = KNeighborsRegressor(n_neighbors, weights=weights)
        else:
            self.model = KNeighborsClassifier(n_neighbors, weights=weights)

    def fit(self, X, y, **kwargs):
        present_categoricals_cols = list(set(X.columns).intersection(set(self._categorical_features)))
        target_enc_values = self._target_enc.fit_transform(X[present_categoricals_cols], y)
        X = X.drop(columns=present_categoricals_cols)
        X[present_categoricals_cols] = target_enc_values
        if self._min_max_scaling:
            X = self._minmax_scaler.fit_transform(X)
        self.model.fit(X, y)
        return self

    def predict(self, X):
        present_categoricals_cols = list(set(X.columns).intersection(set(self._categorical_features)))
        target_enc_values = self._target_enc.transform(X[present_categoricals_cols])
        X = X.drop(columns=present_categoricals_cols)
        X[present_categoricals_cols] = target_enc_values
        if self._min_max_scaling:
            X = self._minmax_scaler.transform(X)
        y_pred = self.model.predict(X)
        return y_pred