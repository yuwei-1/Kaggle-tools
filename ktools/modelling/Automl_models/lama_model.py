# from functools import reduce
# import os
# import random
# from typing import Any, List, Union
# import torch
# from abc import ABC, abstractmethod
# from lightautoml.automl.presets.tabular_presets import TabularAutoML
# from lightautoml.tasks import Task
# from ktools.hyperparameter_optimization.i_sklearn_kfold_object import ISklearnKFoldObject
# from ktools.modelling.Interfaces.i_automl_wrapper import IAutomlWrapper
# from ktools.utils.data_science_pipeline_settings import DataSciencePipelineSettings
# from ktools.preprocessing.basic_feature_transformers import *


# class KToolsLAMAWrapper(IAutomlWrapper):

#     def __init__(self,
#                  train_csv_path : str,
#                  test_csv_path : str,
#                  target_col_name : str,
#                  kfold_object : ISklearnKFoldObject,
#                  task : str = "reg",
#                  metric : str = "mse",
#                  time_limit : float = 3600,
#                  verbosity : int = 2,
#                  lama_models : List[List[str]] = [['lgb', 'lgb_tuned', 'cb', 'cb_tuned']],
#                  data_transforms : List[Any] = [FillNullValues.transform,
#                                                 ConvertObjectToCategorical.transform],
#                  model_name : Union[str, None] = None,
#                  random_state : int = 42,
#                  save_predictions : bool = True,
#                  save_path : str = ""
#                  ) -> None:
#         self._task = task
#         self._metric = metric
#         self._time_limit = time_limit
#         self._verbosity = verbosity
#         self._lama_models = lama_models

#         super().__init__(train_csv_path,
#                          test_csv_path,
#                          target_col_name,
#                          kfold_object,
#                          data_transforms,
#                          model_name,
#                          random_state,
#                          save_predictions,
#                          save_path
#                          )

#     def _set_model_name_and_save_paths(self, model_name):
#         self._model_name = model_name if model_name is not None else '_'.join(self._lama_models[0])
#         self._oof_save_path = os.path.join(self._save_path, f"{model_name}_lama_oof.csv")
#         self._test_save_path = os.path.join(self._save_path, f"{model_name}_lama_test.csv")

#     def _model_setup(self) -> TabularAutoML:

#         task = Task(self._task, metric=self._metric)
#         predictor = TabularAutoML(
#             task = task,
#             timeout = self._time_limit,
#             general_params={"use_algos": self._lama_models})

#         return predictor

#     def fit(self):
#         X, y = self.train_df.drop(columns=self._target_col_name), self.train_df[[self._target_col_name]]
#         roles = {'target' : self._target_col_name}
#         oof_pred = self.model.fit_predict(self.train_df,
#                                           roles = roles,
#                                           verbose = 2,
#                                           cv_iter=list(self._kfold_object.split(X, y))
#                                           )
#         self.oof_pred = pd.Series(oof_pred.data)
#         return self

#     def predict(self, df : Union[pd.DataFrame, None] = None):
#         if df is not None:
#             all_y_preds = self.model.predict(df)
#             all_y_preds = pd.Series(all_y_preds.data)
#             if self._save_predictions: all_y_preds.to_csv(self._test_save_path)
#         else:
#             all_y_preds = self.oof_pred
#             if self._save_predictions: all_y_preds.to_csv(self._oof_save_path)
#         return all_y_preds
