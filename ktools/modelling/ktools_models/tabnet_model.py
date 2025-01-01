from typing import List
import pandas as pd
import math
from sklearn.model_selection import train_test_split
from pytorch_tabnet.tab_model import TabNetClassifier, TabNetRegressor
from ktools.modelling.Interfaces.i_sklearn_model import ISklearnModel


class TabNetModel(ISklearnModel):

    def __init__(self,
                 cat_idcs : List[int],
                 cat_dims : List[int],
                 cat_emb_dims : List[int] = None,
                 eval_metric : List[str] = None,
                 batch_size : int = 1024,
                 virtual_batch_size : int = 128,
                 num_workers : int = 0,
                 drop_last : bool = False,
                 max_epochs : int = 200,
                 patience : int = 50,
                 random_state=129,
                 verbose=0,
                 seed=0,
                 task : str = "regression",
                 **tabnet_params) -> None:
        super().__init__()

        cat_emb_dims = cat_emb_dims if cat_emb_dims is not None else [int(math.sqrt(x)) for x in cat_dims]
        self._batch_size = batch_size
        self._virtual_batch_size = virtual_batch_size
        self._num_workers = num_workers
        self._drop_last = drop_last
        self._max_epochs = max_epochs
        self._patience = patience
        self._random_state = random_state
        self._verbose = verbose
        self._seed = seed
        self._task = task

        if task == "binary":
            self.model = TabNetClassifier(
                                        cat_idxs=cat_idcs,
                                        cat_dims=cat_dims,
                                        cat_emb_dim=cat_emb_dims,
                                        verbose=verbose,
                                        seed=seed,
                                        **tabnet_params
                                        )
            self._eval_metric = eval_metric if eval_metric is not None else ['auc']

        elif task == "regression":
            self.model = TabNetRegressor(
                                        cat_idxs=cat_idcs,
                                        cat_dims=cat_dims,
                                        cat_emb_dim=cat_emb_dims,
                                        verbose=verbose,
                                        seed=seed,
                                        **tabnet_params
                                        )
            self._eval_metric = eval_metric if eval_metric is not None else ['rmse']
    
    def fit(self, X, y, validation_set = None, val_size=0.05):
        if validation_set is None:
            X_train, X_valid, y_train, y_valid = train_test_split(X, 
                                                                  y, 
                                                                  test_size=val_size, 
                                                                  random_state=self._random_state)
        else:
            X_train, y_train = X, y
            X_valid, y_valid = validation_set

        X_train, X_valid, y_train, y_valid = X_train.values, X_valid.values, y_train.values.squeeze(), y_valid.values.squeeze()
        self.model.fit(
                        X_train=X_train, y_train=y_train,
                        eval_set=[(X_valid, y_valid)],
                        eval_name=['val'],
                        eval_metric=self._eval_metric,  
                        batch_size=self._batch_size,
                        virtual_batch_size=self._virtual_batch_size,
                        num_workers=self._num_workers,
                        drop_last=self._drop_last,
                        max_epochs=self._max_epochs,
                        patience = self._patience,
                    )

        return self

    def predict(self, X : pd.DataFrame):
        X = X.values
        if self._task == "regression":
            y_pred = self.model.predict(X)
        elif self._task == "binary":
            y_pred = self.model.predict_proba(X)
        return y_pred