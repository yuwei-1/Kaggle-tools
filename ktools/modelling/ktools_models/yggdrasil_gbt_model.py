from ydf import GradientBoostedTreesLearner, Task
from typing import List, Union
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from ktools.modelling.Interfaces.i_sklearn_model import ISklearnModel


class YDFGBoostModel(ISklearnModel):

    target_col = "target"

    def __init__(self,
                 num_boost_round : int = 100,
                 early_stopping_rounds : int = 20,
                 task : str = "REGRESSION",
                 categorical_algorithm : str = "RANDOM",
                 grow_strategy : str = "BEST_FIRST_GLOBAL",
                 loss : str = "SQUARED_ERROR",
                 random_state : int = 42,
                 verbose : bool = False,
                 **model_kwargs
                 ) -> None:
        super().__init__()
        self._random_state = random_state
        self._verbose = verbose

        task = Task.CLASSIFICATION if task.upper() == "CLASSIFICATION" else Task.REGRESSION
        self.model = GradientBoostedTreesLearner(label = self.target_col,
                                                 task = task,
                                                 categorical_algorithm = categorical_algorithm,
                                                 growing_strategy = grow_strategy,
                                                 loss = loss,
                                                 early_stopping_num_trees_look_ahead = early_stopping_rounds,
                                                 num_trees = num_boost_round,
                                                 **model_kwargs)

    def _convert_back_to_dataset(self, X, y):
        X[self.target_col] = y.values
        return X
    
    def fit(self, X : pd.DataFrame, y : Union[pd.DataFrame, pd.Series, np.ndarray], 
            validation_set = None, val_size=0.05):
        
        if validation_set is None:
            X_train, X_valid, y_train, y_valid = train_test_split(X, 
                                                                  y, 
                                                                  test_size=val_size, 
                                                                  random_state=self._random_state)
        else:
            X_train, y_train = X, y
            X_valid, y_valid = validation_set
        
        train_df = self._convert_back_to_dataset(X_train, y_train)
        valid_df = self._convert_back_to_dataset(X_valid, y_valid)

        self.model = self.model.train(train_df, valid=valid_df, verbose=self._verbose)
        return self

    def predict(self, X):
        y_pred = self.model.predict(X)
        return y_pred