import pandas as pd
import numpy as np
from sklearn.model_selection import RepeatedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import TargetEncoder, StandardScaler
from sklearn.svm import SVR, SVC
from ktools.modelling.Interfaces.i_ktools_model import IKtoolsModel


class SVMModel(IKtoolsModel):

    def __init__(self, 
                 categorical_features, 
                 smooth="auto", 
                 target_type="continuous",
                 standard_scaling=False,
                 **svm_params) -> None:

        self._target_enc = TargetEncoder(target_type=target_type, smooth=smooth)
        self._categorical_features = categorical_features
        self._standard_scaling = standard_scaling
        self._standard_scaler = StandardScaler()

        if target_type == "continuous":
            self.model = SVR(**svm_params)
        else:
            self.model = SVC(**svm_params)

    def fit(self, X, y, **kwargs):
        target_enc_values = self._target_enc.fit_transform(X[self._categorical_features], y)
        X = X.drop(columns=self._categorical_features)
        X[self._categorical_features] = target_enc_values
        if self._standard_scaler:
            X = self._standard_scaler.fit_transform(X)
        self.model.fit(X, y)
        return self

    def predict(self, X):
        target_enc_values = self._target_enc.transform(X[self._categorical_features])
        X = X.drop(columns=self._categorical_features)
        X[self._categorical_features] = target_enc_values
        if self._standard_scaling:
            X = self._standard_scaler.transform(X)
        y_pred = self.model.predict(X)
        return y_pred