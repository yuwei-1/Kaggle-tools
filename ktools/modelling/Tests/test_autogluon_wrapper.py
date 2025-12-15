import unittest
import pandas as pd
from sklearn.model_selection import KFold
from ktools.modelling.Automl_models.autogluon_model import KToolsAutogluonWrapper


class TestKToolsAutogluonWrapper(unittest.TestCase):
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
        ktools_ag_model = KToolsAutogluonWrapper(
            train_csv_path,
            test_csv_path,
            target_col_name,
            kf,
            eval_metric="root_mean_squared_error",
            problem_type="regression",
            fit_kwargs={
                "verbosity": 2,
                "num_cpus": 1,
                "num_gpus": 0,
                "presets": "best_quality",
                "time_limit": 60,
            },
            save_predictions=False,
            save_path="./ktools/modelling/Tests/TestData",
        ).fit()

        oof_pred = ktools_ag_model.predict()
        self.assertIsInstance(oof_pred, pd.DataFrame)

        # pd.testing.assert_frame_equal(pd.read_csv("ktools/modelling/Tests/TestData/expected_ag_output.csv", index_col=0),
        #                               oof_pred, check_dtype=False)
