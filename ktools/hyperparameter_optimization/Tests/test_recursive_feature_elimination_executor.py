import unittest
from copy import deepcopy
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import KFold
from ktools.fitting.cross_validation_executor import CrossValidationExecutor
from ktools.hyperparameter_optimization.recursive_feature_elimination_executor import RecursiveFeatureEliminationExecutor
from ktools.modelling.ktools_models.lgbm_model import LGBMModel
from ktools.preprocessing.basic_feature_transformers import FillNullValues
from ktools.utils.data_science_pipeline_settings import DataSciencePipelineSettings



class TestRecursiveFeatureEliminationExecutor(unittest.TestCase):

    def setUp(self) -> None:
        train_csv_path = "/Users/yuwei-1/Documents/projects/Kaggle-tools/data/used_car_prices/train.csv"
        test_csv_path = "/Users/yuwei-1/Documents/projects/Kaggle-tools/data/used_car_prices/test.csv"
        target_col_name = "price"

        settings = DataSciencePipelineSettings(train_csv_path,
                                                test_csv_path,
                                                target_col_name)
        
        settings = FillNullValues.transform(settings)
        train_df, test_df = settings.update()

        train_df[settings.categorical_col_names] = train_df[settings.categorical_col_names].astype("category")
        self.cat_cols = settings.categorical_col_names
        self.train_cols = settings.training_col_names
        self.X, self.y = train_df.drop(columns=target_col_name), train_df[target_col_name]

        model = LGBMModel()
        kf = KFold(n_splits=5, shuffle=True, random_state=123)
        cve = CrossValidationExecutor(model,
                                      root_mean_squared_error,
                                      kf,
                                      use_test_as_valid=True
                                      )
        
        self.rfecv = RecursiveFeatureEliminationExecutor(cve,
                                                         self.train_cols,
                                                         verbose=True)
    
    def test_rfecv(self):
       best_feature_set, best_oof, best_score =  self.rfecv.run(self.X, self.y)

 
       self.assertEqual(best_feature_set, ['brand',
                                           'model_year', 
                                           'milage', 
                                           'fuel_type', 
                                           'engine', 
                                           'transmission', 
                                           'ext_col', 
                                           'accident'])
       
       self.assertEqual(best_score, 73104.67688185316)