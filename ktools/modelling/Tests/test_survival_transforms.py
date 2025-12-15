import unittest
import pandas as pd
import matplotlib.pyplot as plt
from ktools.modelling.model_transform_wrappers.survival_model_wrapper import *
from post_HCT_survival_notebooks.hct_utils import score


class TestSurvivalTransform(unittest.TestCase):
    def setUp(self) -> None:
        self.train_csv_path = "data/post_hct_survival/train.csv"

        def scci_metric(
            y_test,
            y_pred,
            id_col_name: str = "ID",
            survived_col_name: str = "efs",
            survival_time_col_name: str = "efs_time",
            stratify_col_name: str = "race_group",
        ):
            idcs = y_test.index
            og_train = pd.read_csv(self.train_csv_path)

            y_true = og_train.loc[
                idcs,
                [
                    id_col_name,
                    survived_col_name,
                    survival_time_col_name,
                    stratify_col_name,
                ],
            ].copy()
            y_pred_df = og_train.loc[idcs, [id_col_name]].copy()
            y_pred_df["prediction"] = y_pred
            scci = score(y_true.copy(), y_pred_df.copy(), id_col_name)
            return scci

        self.scci_metric = scci_metric
        self.train_df = pd.read_csv(self.train_csv_path)

    def test_transform_kaplan_meier(self):
        transform = transform_kaplan_meier(
            self.train_df["efs_time"], self.train_df["efs"]
        )
        metric_value = self.scci_metric(self.train_df, transform)
        print(metric_value)
        plt.figure()
        plt.hist(transform, bins=100)
        plt.xlim(0, 1)
        plt.show()
        self.assertTrue(metric_value > 0.5)

    def test_transform_kaplan_meier_with_separation(self):
        transform = transform_kaplan_meier(
            self.train_df["efs_time"], self.train_df["efs"], separation=0.3
        )
        metric_value = self.scci_metric(self.train_df, transform)

        yes_event = self.train_df.efs.values == 1
        plt.figure()
        plt.hist(transform[yes_event], bins=100, color="orange", label="Patient died")
        plt.hist(
            transform[~yes_event],
            bins=100,
            color="blue",
            label="Patient death not observed",
        )
        plt.xlim(0, 1)
        plt.legend()
        plt.show()
        self.assertTrue(metric_value > 0.5)

    def test_transform_weibull(self):
        transform = transform_weibull(self.train_df["efs_time"], self.train_df["efs"])
        metric_value = self.scci_metric(self.train_df, transform)
        print(metric_value)
        plt.figure()
        plt.hist(transform, bins=100)
        plt.xlim(0, 1)
        plt.show()
        self.assertTrue(metric_value > 0.5)

    def test_transform_lognormal(self):
        transform = transform_log_normal(
            self.train_df["efs_time"], self.train_df["efs"]
        )
        metric_value = self.scci_metric(self.train_df, transform)
        print(metric_value)
        plt.figure()
        plt.hist(transform, bins=100)
        plt.xlim(0, 1)
        plt.show()
        self.assertTrue(metric_value > 0.5)

    def test_transform_loglogistic(self):
        transform = transform_log_logistic(
            self.train_df["efs_time"], self.train_df["efs"]
        )
        metric_value = self.scci_metric(self.train_df, transform)
        print(metric_value)
        plt.figure()
        plt.hist(transform, bins=100)
        plt.xlim(0, 1)
        plt.show()
        self.assertTrue(metric_value > 0.5)

    def test_transform_nelson_aalen(self):
        transform = transform_nelson_aalen(
            self.train_df["efs_time"], self.train_df["efs"]
        )
        metric_value = self.scci_metric(self.train_df, transform)
        print(metric_value)
        plt.figure()
        plt.hist(transform, bins=100)
        plt.show()
        self.assertTrue(metric_value > 0.5)

    def test_transform_partial_hazard(self):
        transform = transform_partial_hazard(
            self.train_df["efs_time"], self.train_df["efs"]
        )
        metric_value = self.scci_metric(self.train_df, transform)
        print(metric_value)
        plt.figure()
        plt.hist(transform, bins=100)
        plt.show()
        self.assertTrue(metric_value > 0.5)

    def test_transform_separate(self):
        transform = transform_separate(self.train_df["efs_time"], self.train_df["efs"])
        metric_value = self.scci_metric(self.train_df, transform)
        print(metric_value)
        plt.figure()
        plt.hist(transform, bins=100)
        plt.show()
        self.assertTrue(metric_value > 0.5)

    def test_transform_rank_log(self):
        transform = transform_rank_log(self.train_df["efs_time"], self.train_df["efs"])
        metric_value = self.scci_metric(self.train_df, transform)
        print(metric_value)
        plt.figure()
        plt.hist(transform, bins=100)
        plt.show()
        self.assertTrue(metric_value > 0.5)

    def test_transform_quantile(self):
        transform = transform_quantile(self.train_df["efs_time"], self.train_df["efs"])
        metric_value = self.scci_metric(self.train_df, transform)
        print(metric_value)
        plt.figure()
        plt.hist(transform, bins=100)
        plt.show()
        self.assertTrue(metric_value > 0.5)

    def test_transform_aalen_johansen(self):
        transform = transform_aalen_johansen(
            self.train_df["efs_time"], self.train_df["efs"]
        )
        metric_value = self.scci_metric(self.train_df, transform)
        print(metric_value)
        plt.figure()
        plt.hist(transform, bins=100)
        plt.show()
        self.assertTrue(metric_value > 0.5)

    def test_transform_breslow_flemingharrington(self):
        transform = transform_breslow_flemingharrington(
            self.train_df["efs_time"], self.train_df["efs"]
        )
        metric_value = self.scci_metric(self.train_df, transform)
        print(metric_value)
        plt.figure()
        plt.hist(transform, bins=100)
        plt.show()
        self.assertTrue(metric_value > 0.5)

    def test_transform_exponential(self):
        transform = transform_exponential(
            self.train_df["efs_time"], self.train_df["efs"]
        )
        metric_value = self.scci_metric(self.train_df, transform)
        print(metric_value)
        plt.figure()
        plt.hist(transform, bins=100)
        plt.show()
        self.assertTrue(metric_value > 0.5)

    def test_transform_generalized_gamma(self):
        transform = transform_generalized_gamma(
            self.train_df["efs_time"], self.train_df["efs"]
        )
        metric_value = self.scci_metric(self.train_df, transform)
        print(metric_value)
        plt.figure()
        plt.hist(transform, bins=100)
        plt.show()
        self.assertTrue(metric_value > 0.5)
