
import lightgbm as lgb
from lightgbm import early_stopping, log_evaluation
from ktools.fitting.i_sklearn_model import ISklearnModel
from sklearn.model_selection import train_test_split


class LGBMModel(ISklearnModel):

    def __init__(self,
                 log_period=150,
                 stopping_rounds=200,
                 **lgb_param_grid,) -> None:
        super().__init__()
        self._lgb_param_grid = {**lgb_param_grid}
        self._callbacks = [log_evaluation(period=log_period), 
                           early_stopping(stopping_rounds=stopping_rounds)]
        
    def fit(self, X, y, val_size=0.05, random_state=129):
        X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=val_size, random_state=random_state)
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_valid, label=y_valid, reference=train_data)
        self.model = lgb.train(self._lgb_param_grid,
                                train_data,
                                valid_sets=[train_data, val_data],
                                valid_names=['train', 'valid'],
                                callbacks=self._callbacks,
                                )
        return self.model

    def predict(self, X):
        y_pred = self.model.predict(X)
        return y_pred