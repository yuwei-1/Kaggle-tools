import unittest
from sklearn.model_selection import KFold
from ktools.modelling.Automl_models.flaml_model import KToolsFLAMLWrapper
from ktools.preprocessing.basic_feature_transformers import *


class TestKToolsFLAMLWrapper(unittest.TestCase):
    def test_basic_usage(self):
        train_csv_path = (
            "/Users/yuwei-1/Documents/projects/Kaggle-tools/data/insurance/train.csv"
        )
        test_csv_path = (
            "/Users/yuwei-1/Documents/projects/Kaggle-tools/data/insurance/test.csv"
        )
        sample_sub_csv_path = "/Users/yuwei-1/Documents/projects/Kaggle-tools/data/insurance/sample_submission.csv"
        target_col_name = "Premium Amount"

        kf = KFold(5, shuffle=True, random_state=42)
        ktools_ag_model = KToolsFLAMLWrapper(
            train_csv_path,
            test_csv_path,
            target_col_name,
            kf,
            model_name="flaml",
            eval_metric="mse",
            data_transforms=[
                FillNullValues.transform,
                ConvertObjectToCategorical.transform,
                LogTransformTarget.transform,
            ],
            total_time_budget=300,
            ensemble=True,
            ensemble_size=5,
            verbose=3,
            save_predictions=False,
            save_path="./ktools/modelling/Tests/TestData",
        ).fit()

        # ktools_ag_model = ktools_ag_model.fit()
        oof_pred = ktools_ag_model.predict()

        # self.assert
