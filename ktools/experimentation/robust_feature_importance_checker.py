from functools import reduce
import os
from scipy.stats import ks_2samp
from typing import Dict, List
from ktools.fitting.cross_validation_executor import CrossValidationExecutor
from ktools.modelling.ktools_models.catboost_model import CatBoostModel
from ktools.modelling.ktools_models.lgbm_model import LGBMModel
from ktools.feature_engineering.interfaces.i_feature_creator import IFeatureCreator
from ktools.hyperparameter_optimization.i_sklearn_kfold_object import ISklearnKFoldObject
from ktools.modelling.Interfaces.i_sklearn_model import ISklearnModel
from ktools.preprocessing.basic_feature_transformers import *
from ktools.utils.data_science_pipeline_settings import DataSciencePipelineSettings



class RobustFeatureImportanceChecker:

    def __init__(self,
                 train_csv_path : str,
                 test_csv_path : str,
                 target_col_name : str,
                 feature_creator,
                 kfold_object,
                 metric_callable : callable,
                 metric_direction : str = "maximize",
                 model_type : str = 'lgbm',
                 model_params : Dict = {},
                 sqrt_population_size : int = 10,
                 result_path : str = None,
                 initial_transform_list : List[callable] = [FillNullValues.transform,
                                                            ConvertObjectToCategorical.transform],
                 original_csv_path : str = None
                 ) -> None:
        self._train_csv_path = train_csv_path
        self._test_csv_path = test_csv_path
        self._target_col_name = target_col_name
        self._feature_creator = feature_creator
        self._kfold_object = kfold_object
        self._metric_callable = metric_callable
        self._metric_direction = metric_direction
        self._model_type = model_type
        self._model_params = model_params
        self._sqrt_population_size = sqrt_population_size
        self._result_path = result_path
        self._initial_transform_list = initial_transform_list
        self._original_csv_path = original_csv_path
        combined_df, self._added_feature_names, self._original_col_names = self._setup()
        self._train_df, self._test_df = combined_df.loc['train'], combined_df.loc['test']


    def _setup(self):
        settings = DataSciencePipelineSettings(self._train_csv_path,
                                                self._test_csv_path,
                                                self._target_col_name,
                                                original_csv_path=self._original_csv_path
                                                )
        settings = reduce(lambda acc, func: func(acc), self._initial_transform_list, settings)
        original_col_names = settings.training_col_names
        settings.update()
        combined_df, added_feature_names = self._feature_creator.create(settings.combined_df)
        # combined_df.to_csv("combined_df.csv")
        return combined_df, added_feature_names, original_col_names
    
    def run(self):
        feature_col_name = "added_feature"
        score_col_name = "cv_score"
        significance_col_name = "significance"
        importance_col_name = "important_feature"

        X, y = self._train_df.drop(columns=self._target_col_name), self._train_df[self._target_col_name]
        initial_score_population_path = os.path.join(self._result_path, 'initial_score_population.npy')
        csv_results_path = os.path.join(self._result_path, 'robust_feature_importance.csv')
        history =  pd.read_csv(csv_results_path) if os.path.exists(csv_results_path) else pd.DataFrame({feature_col_name : [], 
                                                                                                        score_col_name : [], 
                                                                                                        significance_col_name : [],
                                                                                                        importance_col_name : []})
        if os.path.exists(initial_score_population_path):
            print("Initial population found")
            initial_score_population = np.load(initial_score_population_path)
        else:
            print("#"*100)
            print("Initial population not found, creating now...")
            print("#"*100)
            all_scores = []
            for model_random_state in range(42, 42+self._sqrt_population_size):
                for cv_random_state in range(42, 42+self._sqrt_population_size):
                    model = self.get_model_instance(model_random_state)
                    self._kfold_object.random_state = cv_random_state
                    score_tuple,_,_ = CrossValidationExecutor(model,
                                                                self._metric_callable,
                                                                self._kfold_object,
                                                                ).run(X[self._original_col_names], y)
                    all_scores += [score_tuple[0]]
            initial_score_population = np.array(all_scores)
            np.save(initial_score_population_path, initial_score_population)
        
        if not (history[feature_col_name] == "original").any():
            new_entry = pd.DataFrame({feature_col_name : ["original"], 
                                      score_col_name : [initial_score_population.mean()], 
                                      significance_col_name : [0],
                                      importance_col_name : [None]
                                      })
            history = pd.concat([history, new_entry])
            history.to_csv(csv_results_path, index=False)
        
        for feature in self._added_feature_names:
            all_scores = []
            for model_random_state in range(1024, 1024+int(self._sqrt_population_size*0.5)):
                for cv_random_state in range(1024, 1024+int(self._sqrt_population_size*0.5)):
                    model = self.get_model_instance(model_random_state)
                    self._kfold_object.random_state = cv_random_state
                    score_tuple,_,_ = CrossValidationExecutor(model,
                                                                self._metric_callable,
                                                                self._kfold_object,
                                                                ).run(X[self._original_col_names + [feature]], y)
                    all_scores += [score_tuple[0]]

            feature_subsample = np.array(all_scores)
            res = ks_2samp(initial_score_population, feature_subsample)

            significance = 0.05
            print("#"*100)
            print("RESULT: ", res)
            print(f"Original mean: {initial_score_population.mean()}, New mean: {feature_subsample.mean()}")
            correct_direction = (initial_score_population.mean() < feature_subsample.mean()) if self._metric_direction == "maximize" else (initial_score_population.mean() > feature_subsample.mean())
            important = (res.pvalue < significance) & correct_direction
            print("CHANGE IS USEFUL: ", important)
            print("#"*100)


            new_entry = pd.DataFrame({feature_col_name : [feature], 
                                      score_col_name : [feature_subsample.mean()], 
                                      significance_col_name : [res.pvalue],
                                      importance_col_name : [important]
                                      })
            history = pd.concat([history, new_entry])
            history.to_csv(csv_results_path, index=False)
    
    def get_model_instance(self, random_state):
        if self._model_type == 'lgbm':
            return LGBMModel(random_state=random_state, colsample_bytree=0.9, subsample=0.9, **self._model_params)
        elif self._model_type == 'cat':
            return CatBoostModel(random_state=random_state, subsample=0.9, colsample_bylevel=0.9, **self._model_params)
        else:
            raise NotImplementedError