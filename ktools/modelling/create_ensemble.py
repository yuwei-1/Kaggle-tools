import json
import os
import sys
from typing import Dict
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
sys.path.append("/Users/yuwei-1/Documents/projects/Kaggle-tools")
from scipy.optimize import minimize
from ktools.modelling.ktools_models.catboost_model import CatBoostModel
from ktools.modelling.ktools_models.lgbm_model import LGBMModel
from ktools.modelling.ktools_models.xgb_model import XGBoostModel
from sklearn.metrics import root_mean_squared_error


class CreateFlatEnsemble:

    def __init__(self, 
                 models_json_path : str,
                 kf : KFold,
                 num_splits : int,
                 eval_metric : callable,
                 output_file_path : str,
                 overwrite : bool = False) -> None:
        self._models_json_path = models_json_path
        self._kf = kf
        self._num_splits = num_splits
        self._eval_metric = eval_metric
        self._output_file_path = output_file_path
        self._overwrite = overwrite
        self._model_cls_list, self._param_set_list, self._model_ids = self._load_models()
        self._num_models = len(self._model_cls_list)

    def _load_models(self):
        with open(self._models_json_path, 'r') as file:
            ensemble_settings = json.load(file)
        model_list = ensemble_settings['ensemble_info']
        model_cls_list, param_set_list, model_ids = [], [], []

        for i, model_settings in enumerate(model_list):
            param_set = model_settings.get("params")
            model_type = model_settings.get("model_name")
            model_id = model_settings.get("id")
            model_cls = self._get_model_using_type(model_type)
            model_cls_list += [model_cls]
            param_set_list += [param_set]
            model_ids += [model_id]
        return model_cls_list, param_set_list, model_ids
    
    def _fit_then_predict(self, X, y, X_test, model_idx : int):
        model = self._model_cls_list[model_idx](**self._param_set_list[model_idx]).fit(X, y)
        y_pred = model.predict(X_test)
        return y_pred, model
    
    def create_model_predictions(self, X, y, X_test):
        oof_predictions = np.zeros((self._num_models, X.shape[0]))
        test_predictions = np.zeros((self._num_models, X_test.shape[0]))

        for i, (train_index, val_index) in enumerate(self._kf.split(X, y)):
            X_train_fold, X_val_fold = X.iloc[train_index], X.iloc[val_index]
            y_train_fold, _ = y.iloc[train_index], y.iloc[val_index]

            for midx in range(self._num_models):
                y_pred, model = self._fit_then_predict(X_train_fold,
                                                        y_train_fold,
                                                        X_val_fold,
                                                        model_idx=midx)
                test_predictions[midx] += model.predict(X_test)/self._num_splits
                oof_predictions[midx, val_index] = y_pred

        for midx in range(self._num_models):
            oof_score = self._eval_metric(y.to_numpy().squeeze(), oof_predictions[midx])
            identifier = self._model_ids[midx] + "_" + str(round(oof_score, 2))
            model_dir = os.path.join(self._output_file_path, identifier)
            oof_prediction_df = pd.DataFrame(data=oof_predictions[midx], columns=['oof_predictions'], index=X.index)
            test_prediction_df = pd.DataFrame(data=test_predictions[midx], columns=['test_prediction'], index=X_test.index)
            if not os.path.exists(model_dir):
                os.mkdir(model_dir)
                oof_prediction_df.to_csv(model_dir + "/oof_predictions.csv")
                test_prediction_df.to_csv(model_dir + "/test_predictions.csv")
            else:
                if self._overwrite:
                    oof_prediction_df.to_csv(model_dir + "/oof_predictions.csv")
                    test_prediction_df.to_csv(model_dir + "/test_predictions.csv")

        return True
            # self._output_file_path

        # def objective(weights):
        #     weighted_oof_predictions = weights.reshape(-1, 1) * oof_predictions
        #     loss = root_mean_squared_error(y.to_numpy().squeeze(), weighted_oof_predictions.sum(axis=0))
        #     print("Current weights: ", weights)
        #     print("current loss :", loss)
        #     return loss
        
        # constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        # bounds = [(0, 1)] * self._num_models
        # initial_weights = [1/self._num_models] * self._num_models
        # result = minimize(objective, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)
        # blend_weights = result.x
        # return (blend_weights.reshape(-1, 1)*self.test_predictions).sum(axis=0)

    def _get_model_using_type(self, model_string):
        if model_string == "lgbm":
            return LGBMModel
        elif model_string == "xgb":
            return XGBoostModel
        elif model_string == "cat":
            return CatBoostModel


if __name__ == "__main__":
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    ensemble = CreateFlatEnsemble("/Users/yuwei-1/Documents/projects/Kaggle-tools/ktools/modelling/Data/test_ensemble.json", 
                                  kf, 
                                  10,
                                  root_mean_squared_error,
                                  "/Users/yuwei-1/Documents/projects/Kaggle-tools/data/used_car_prices/oof_test_predictions",
                                  )

    train_df = pd.read_csv("/Users/yuwei-1/Documents/projects/Kaggle-tools/data/used_car_prices/basic_train.csv", index_col=0)
    test_df = pd.read_csv("/Users/yuwei-1/Documents/projects/Kaggle-tools/data/used_car_prices/basic_test.csv", index_col=0)

    cat_cols = [col_name for col_name in train_df.columns if train_df[col_name].dtype == 'object']

    train_df[cat_cols] = train_df[cat_cols].astype('category')
    test_df[cat_cols] = test_df[cat_cols].astype('category')

    test_pred = ensemble.create_model_predictions(train_df.drop(columns="price"), 
                                                    train_df["price"], 
                                                    test_df.drop(columns="price"),
                                                    )