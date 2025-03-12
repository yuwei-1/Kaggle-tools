import unittest
from ktools.utils.data_science_pipeline_settings import DataSciencePipelineSettings


class TestDataSciencePipelinePropertySetting(unittest.TestCase):

    def test_setting_target_col(self):

        train_csv_path = "/Users/yuwei-1/Documents/projects/Kaggle-tools/data/loan_approval/train.csv"
        test_csv_path = "/Users/yuwei-1/Documents/projects/Kaggle-tools/data/loan_approval/test.csv"
        self.target_col_name = "loan_status"

        settings = DataSciencePipelineSettings(train_csv_path,
                                               test_csv_path,
                                               self.target_col_name, 
                                               )
        
        self.assertEqual(settings.target_col, self.target_col_name)

        settings.target_col = "none"

        self.assertEqual(settings.target_col, "none")