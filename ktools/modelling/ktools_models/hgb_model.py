from sklearn.preprocessing import MinMaxScaler, TargetEncoder
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from ktools.modelling.Interfaces.i_sklearn_model import ISklearnModel


class HGBModel(ISklearnModel):

    def __init__(self,
                 smooth="auto", 
                 target_type="continuous",
                 num_boost_round=100,
                 early_stopping=True,
                 validation_fraction=0.05,
                 early_stopping_rounds=20,
                 verbose=0,
                 random_state=129,
                 **hgb_params) -> None:
        hgb_params = {"max_iter" : num_boost_round,
                      "early_stopping" : early_stopping,
                      "validation_fraction" : validation_fraction,
                      "n_iter_no_change" : early_stopping_rounds,
                      "verbose" : verbose,
                      "random_state" : random_state,
                      "categorical_features" : "from_dtype",
                      **hgb_params}
        
        self._target_enc = TargetEncoder(target_type=target_type, 
                                         smooth=smooth, 
                                         random_state=random_state)
        self._target_type = target_type
        if target_type == "continuous":
            self.model = HistGradientBoostingRegressor(**hgb_params)
        else:
            self.model = HistGradientBoostingClassifier(**hgb_params)
        
    
    def fit(self, X, y, validation_set=None, **kwargs):
        y = y.squeeze()
        categorical_features = [col_name for col_name in X.columns if X[col_name].dtype == 'category']
        target_enc_values = self._target_enc.fit_transform(X[categorical_features], y)
        X = X.drop(columns=categorical_features)
        X[categorical_features] = target_enc_values
        self.model.fit(X, y)
        return self
    
    def predict(self, X):
        categorical_features = [col_name for col_name in X.columns if X[col_name].dtype == 'category']
        target_enc_values = self._target_enc.transform(X[categorical_features])
        X = X.drop(columns=categorical_features)
        X[categorical_features] = target_enc_values
        if self._target_type == "continuous":
            y_pred = self.model.predict(X)
        else:
            y_pred = self.model.predict_proba(X)[:, 1]
        return y_pred