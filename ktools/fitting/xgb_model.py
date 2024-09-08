from sklearn.model_selection import train_test_split
import xgboost as xgb
from ktools.fitting.i_sklearn_model import ISklearnModel


class XGBoostModel(ISklearnModel):

    def __init__(self,
                 eval_verbosity=False,
                 stopping_rounds=200,
                 **xgb_param_grid) -> None:
        super().__init__()
        self._eval_verbosity = eval_verbosity
        self._stopping_rounds = stopping_rounds
        self._xgb_param_grid = {**xgb_param_grid}
    
    def fit(self, X, y, val_size=0.05, random_state=129):
        X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=val_size, random_state=random_state)
        train_data = xgb.DMatrix(X_train, label=y_train, enable_categorical=True)
        valid_data = xgb.DMatrix(X_valid, label=y_valid, enable_categorical=True)
        eval_data = [(train_data, 'train'), (valid_data, 'eval')]
    
        self.model = xgb.train(
            self._xgb_param_grid, 
            train_data, 
            evals=eval_data,                       
            early_stopping_rounds=self._stopping_rounds,           
            verbose_eval=self._eval_verbosity                    
        )
        return self.model

    def predict(self, X):
        test_data = xgb.DMatrix(X, enable_categorical=True)
        y_pred = self.model.predict(test_data)
        return y_pred