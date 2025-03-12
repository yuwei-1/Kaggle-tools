import unittest
from sklearn.metrics import r2_score
from parameterized import parameterized
from ktools.modelling.Interfaces.i_ktools_model import IKtoolsModel
from ktools.preprocessing.basic_feature_transformers import ConvertObjectToCategorical
from ktools.utils.data_science_pipeline_settings import DataSciencePipelineSettings
from ktools.modelling.ktools_models.xgb_model import XGBoostModel

class TestKtoolsRegression(unittest.TestCase):
    
    def setUp(self) -> None:
        train_csv_path = "/Users/yuwei-1/Documents/projects/Kaggle-tools/data/loan_approval/train.csv"
        test_csv_path = "/Users/yuwei-1/Documents/projects/Kaggle-tools/data/loan_approval/test.csv"
        self.target_col_name = "loan_status"

        settings = DataSciencePipelineSettings(train_csv_path,
                                               test_csv_path,
                                               self.target_col_name, 
                                               )
        settings = ConvertObjectToCategorical.transform(settings)
        self.X, self.y, self.X_test, _ = settings.get_data()
    
    @parameterized.expand([
        (XGBoostModel(**{"objective" : "reg:squarederror", "eval_metric" : "rmse", "num_boost_round" : 1000})), # Test case 1
    ])
    def test_regression_early_stop(self, model_cls : IKtoolsModel):
        model_cls = model_cls.fit(self.X, self.y)
        y_pred = model_cls.predict(self.X)

        train_score = r2_score(self.y, y_pred)
        num_trained_trees = model_cls.model.num_boosted_rounds()

        self.assertTrue(y_pred.shape == self.y.shape)
        self.assertTrue(num_trained_trees < 1000)
        self.assertTrue(train_score > 0.5)


    @parameterized.expand([
        (XGBoostModel(**{"objective" : "reg:squarederror", "eval_metric" : "rmse", "num_boost_round" : 1000, "early_stopping_rounds" : None})), # Test case 1
    ])
    def test_regression(self, model_cls : IKtoolsModel):
        model_cls = model_cls.fit(self.X, self.y)
        y_pred = model_cls.predict(self.X)

        train_score = r2_score(self.y, y_pred)
        num_trained_trees = model_cls.model.num_boosted_rounds()

        self.assertTrue(y_pred.shape == self.y.shape)
        self.assertTrue(num_trained_trees == 1000)
        self.assertTrue(train_score > 0.5)