import unittest
import pandas as pd
from copy import deepcopy
from functools import reduce
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from ktools.fitting.cross_validation_executor import CrossValidationExecutor
from ktools.modelling.create_oof_from_model import create_oofs_from_model
from ktools.models import LGBMModel
from ktools.preprocessing.basic_feature_transformers import (
    ConvertObjectToCategorical,
    FillNullValues,
)
from ktools.utils.data_science_pipeline_settings import DataSciencePipelineSettings


class TestAdditionalData(unittest.TestCase):
    def setUp(self) -> None:
        train_csv_path = "/Users/yuwei-1/Documents/projects/Kaggle-tools/data/loan_approval/train.csv"
        original_csv_path = "/Users/yuwei-1/Documents/projects/Kaggle-tools/data/loan_approval/original.csv"
        test_csv_path = (
            "/Users/yuwei-1/Documents/projects/Kaggle-tools/data/loan_approval/test.csv"
        )
        self.target_col_name = "loan_status"

        # preprocess without original data first

        settings = DataSciencePipelineSettings(
            train_csv_path,
            test_csv_path,
            self.target_col_name,
        )

        class AddOGData:
            @staticmethod
            def transform(original_settings: DataSciencePipelineSettings):
                settings = deepcopy(original_settings)
                original_data = pd.read_csv(original_csv_path, index_col=0)
                settings.train_df = pd.concat(
                    [settings.train_df.assign(source=0), original_data.assign(source=1)]
                ).reset_index(drop=True)
                settings.combined_df = pd.concat(
                    [settings.train_df, settings.test_df.assign(source=0)],
                    keys=["train", "test"],
                )
                return settings

        class AddLoanPredictionFeatures:
            @staticmethod
            def transform(original_settings: DataSciencePipelineSettings):
                settings = deepcopy(original_settings)
                settings.combined_df["loantoincome"] = (
                    settings.combined_df["loan_amnt"]
                    / settings.combined_df["person_income"]
                ) - settings.combined_df["loan_percent_income"]
                settings.training_col_names += ["loantoincome"]
                settings.categorical_col_names += ["person_income"]
                return settings

        transforms = [
            AddOGData.transform,
            AddLoanPredictionFeatures.transform,
            FillNullValues.transform,
            ConvertObjectToCategorical.transform,
        ]

        settings = reduce(lambda acc, func: func(acc), transforms, settings)
        settings.update()

        self.train_df, self.test_df = settings.update()
        self.test_df.drop(columns=[self.target_col_name], inplace=True)

        # preprocess with original data

        new_settings = DataSciencePipelineSettings(
            train_csv_path,
            test_csv_path,
            self.target_col_name,
            original_csv_path=original_csv_path,
        )

        transforms = [
            AddLoanPredictionFeatures.transform,
            FillNullValues.transform,
            ConvertObjectToCategorical.transform,
        ]

        new_settings = reduce(lambda acc, func: func(acc), transforms, new_settings)
        self.new_train_df, self.new_test_df = new_settings.update()
        self.new_test_df.drop(columns=[self.target_col_name], inplace=True)

    def test_equivalence(self):
        # self.train_df.to_csv("train_csv_check.csv")
        pd.testing.assert_frame_equal(self.train_df, self.new_train_df)
        pd.testing.assert_frame_equal(self.test_df, self.new_test_df)

    # @unittest.skip("Experiment")
    def test_exp_compare_different_data_addition_methods(self):
        sample_sub_csv = "/Users/yuwei-1/Documents/projects/Kaggle-tools/data/loan_approval/sample_submission.csv"

        lgb_params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "num_boost_round": 5000,
            "early_stopping_rounds": 200,
        }
        lgb_model = LGBMModel(**lgb_params)

        kf = StratifiedKFold(10, shuffle=True, random_state=42)

        cve = CrossValidationExecutor(lgb_model, roc_auc_score, kf, verbose=2)

        _ = create_oofs_from_model(
            cve,
            self.train_df.drop(columns=self.target_col_name),
            self.train_df[[self.target_col_name]],
            self.test_df,
            additional_data=None,
            model_string="lgbm_using_previous_method",
            directory_path="",
            sample_submission_file=sample_sub_csv,
        )

        train = self.train_df[self.train_df["source"] == 0]
        original = self.train_df[self.train_df["source"] == 1]

        cve = CrossValidationExecutor(lgb_model, roc_auc_score, kf, verbose=2)

        _ = create_oofs_from_model(
            cve,
            train.drop(columns=self.target_col_name),
            train[[self.target_col_name]],
            self.test_df,
            additional_data=[
                original.drop(columns=self.target_col_name),
                original[[self.target_col_name]],
            ],
            model_string="lgbm_using_additional_as_trainingonly",
            directory_path="",
            sample_submission_file=sample_sub_csv,
        )
