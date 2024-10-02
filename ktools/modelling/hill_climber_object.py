from typing import Dict, Tuple
import pandas as pd
from hillclimbers import climb_hill
from functools import partial


class HillClimber:

    def __init__(self,
                 train_df : pd.DataFrame,
                 train_oof_pred : pd.DataFrame,
                 test_pred : pd.DataFrame,
                 target_col_name : str,
                 eval_metric : callable,
                 objective : str = "minimize",
                 model_prediction_file_paths : Dict[str, Tuple[str]] = None, # train oof, test file paths
                 negative_weights : bool = False,
                 plot_hill : bool = False
                 ) -> None:
        self._train_df = train_df
        self.train_oof_pred = train_oof_pred
        self.test_pred = test_pred
        self._target_col_name = target_col_name
        self._eval_metric = partial(eval_metric)
        self._objective = objective
        self._model_prediction_file_paths = model_prediction_file_paths
        self._negative_weights = negative_weights
        self._plot_hill = plot_hill
        self.load_saved_prediction_files(model_prediction_file_paths)

    def naive_hill_climb(self):

        best_test_pred = climb_hill(train=self._train_df,
                                    oof_pred_df=self.train_oof_pred,
                                    test_pred_df=self.test_pred,
                                    target=self._target_col_name,
                                    objective=self._objective,
                                    eval_metric=self._eval_metric,
                                    negative_weights=self._negative_weights,
                                    plot_hill=self._plot_hill)
        
        return best_test_pred
    
    def load_saved_prediction_files(self, model_prediction_file_paths):
        if model_prediction_file_paths is not None:
            for model_name, (train_oof_file_path, test_pred_file_path) in model_prediction_file_paths.items():

                train_oof = pd.read_csv(train_oof_file_path, index_col=0)
                test_pred = pd.read_csv(test_pred_file_path, index_col=0)

                train_col_name = train_oof.columns[0]
                test_col_name = test_pred.columns[0]

                train_oof.rename(columns={train_col_name : model_name}, inplace=True)
                test_pred.rename(columns={test_col_name : model_name}, inplace=True)

                self.train_oof_pred = pd.concat([self.train_oof_pred, train_oof], axis=1)
                self.test_pred = pd.concat([self.test_pred, test_pred], axis=1)