from functools import reduce
import os
import h2o
from h2o.automl import H2OAutoML
from typing import Any, Dict, List, Union
from ktools.hyperparameter_optimization.i_sklearn_kfold_object import ISklearnKFoldObject
from ktools.modelling.Interfaces.i_automl_wrapper import IAutomlWrapper
from ktools.preprocessing.basic_feature_transformers import *



class KToolsH2OWrapper(IAutomlWrapper):

    def __init__(self,
                 train_csv_path : str,
                 test_csv_path : str,
                 target_col_name : str,
                 kfold_object : ISklearnKFoldObject,
                 data_transforms : List[Any] = [FillNullValues.transform,
                                                ConvertObjectToCategorical.transform],
                 time_limit : float = 3600,
                 algos : List[str] = ["GBM", "DRF", "XGBoost", "DeepLearning"],
                 verbosity : str = "info",
                 model_name : Union[str, None] = None,
                 random_state : int = 42,
                 save_predictions : bool = True,
                 save_path : str = ""
                 ) -> None:
        
        h2o.init()
        self._time_limit = time_limit
        self._algos = algos
        self._verbosity = verbosity
        
        super().__init__(train_csv_path,
                         test_csv_path,
                         target_col_name,
                         kfold_object,
                         data_transforms,
                         model_name,
                         random_state,
                         save_predictions,
                         save_path
                         )

    def _set_model_name_and_save_paths(self, model_name):
        self._model_name = model_name #if model_name is not None else '_'.join(self._included_model_types)
        self._oof_save_path = os.path.join(self._save_path, f"{model_name}_h2o_oof.csv")
        self._test_save_path = os.path.join(self._save_path, f"{model_name}_h2o_test.csv")
    
    def _model_setup(self):
        self.kfname = kfold_col_name = "fold"

        X, y = self.train_df.drop(columns=self._target_col_name), self.train_df[[self._target_col_name]]
        split = self._kfold_object.split(X, y)
        for i, (_, val_index) in enumerate(split):
            self.train_df.loc[val_index, kfold_col_name] = i

        aml = H2OAutoML(
            max_runtime_secs=self._time_limit,
            include_algos=self._algos,
            keep_cross_validation_predictions=True,
            seed=self._random_state,
            verbosity=self._verbosity
        )

        return aml

    def fit(self):
        h_train = h2o.H2OFrame(self.train_df)
        x = [col for col in h_train.columns if col not in [self._target_col_name, self.kfname]]
        self.model.train(x=x, y=self._target_col_name, training_frame=h_train, fold_column=self.kfname)
        leaderboard = self.model.leaderboard.as_data_frame()
        self.model_ids = leaderboard['model_id'].tolist()
        return self
    
    def predict(self, df : Union[pd.DataFrame, None] = None):
        all_y_preds = pd.DataFrame()
        if df is not None:
            h_test = h2o.H2OFrame(df)
            for model_id in self.model_ids:
                model = h2o.get_model(model_id)
                test_predictions = model.predict(h_test).as_data_frame()
                all_y_preds[model_id] = test_predictions['predict']
            if self._save_predictions: all_y_preds.to_csv(self._test_save_path)
        else:
            for model_id in self.model_ids:
                model = h2o.get_model(model_id)
                oof_predictions = model.cross_validation_holdout_predictions().as_data_frame()
                all_y_preds[model_id] = oof_predictions['predict']
            if self._save_predictions: all_y_preds.to_csv(self._oof_save_path)

        return all_y_preds