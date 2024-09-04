import os
import sys
import unittest
import pandas as pd
from ktools.preprocessing.categorical_denoiser_prepreprocesser import CategoricalDenoiserPreprocessor
from ktools.utils.data_science_pipeline_settings import DataSciencePipelineSettings



class TestCategoricalDenoiserPreprocessor(unittest.TestCase):

    def test_process(self):
        current_directory = os.path.dirname(os.path.abspath(__file__))

        expected_denoised_category_training_df = pd.read_csv(os.path.join(current_directory, "TestData/expected_denoised_category_training_data.csv"), index_col=0)
        expected_denoised_category_testing_df = pd.read_csv(os.path.join(current_directory, "TestData/expected_denoised_category_testing_data.csv"), index_col=0)

        train_csv_path = "data/poisonous_mushrooms/train.csv"
        test_csv_path = "data/poisonous_mushrooms/test.csv"
        target_col_name = "class"

        settings = DataSciencePipelineSettings(train_csv_path,
                                                test_csv_path,
                                                target_col_name)

        denoised_category_training_df, denoised_category_testing_df, _ = CategoricalDenoiserPreprocessor(settings).process()
        
        pd.testing.assert_frame_equal(expected_denoised_category_training_df, denoised_category_training_df)
        pd.testing.assert_frame_equal(expected_denoised_category_testing_df, denoised_category_testing_df)