import unittest
import numpy as np
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import KFold
from ktools.fitting.cross_validation_executor import CrossValidationExecutor
from ktools.models import LGBMModel
from ktools.utils.data_science_pipeline_settings import DataSciencePipelineSettings


class TestCrossValidationExecutor(unittest.TestCase):
    def setUp(self) -> None:
        train_csv_path = "/Users/yuwei-1/Documents/projects/Kaggle-tools/data/used_car_prices/train.csv"
        test_csv_path = "/Users/yuwei-1/Documents/projects/Kaggle-tools/data/used_car_prices/test.csv"
        target_col_name = "price"

        settings = DataSciencePipelineSettings(
            train_csv_path, test_csv_path, target_col_name
        )

        train_df, test_df = settings.update()

        train_df[settings.categorical_col_names] = train_df[
            settings.categorical_col_names
        ].astype("category")
        self.X, self.y = (
            train_df.drop(columns=target_col_name),
            train_df[target_col_name],
        )

    def test_cross_validation_outputs(self):
        num_folds = 5
        model = LGBMModel(**{"n_jobs": 1})
        kf = KFold(num_folds, shuffle=True, random_state=42)
        cv_scores, oof, model_list = CrossValidationExecutor(
            model,
            root_mean_squared_error,
            kf,
        ).run(self.X, self.y)

        self.assertEqual(len(cv_scores), 2)
        self.assertEqual(type(cv_scores[0]), np.float64)
        self.assertEqual(len(model_list), 5)
        self.assertEqual(len(oof), len(self.X))

        for i in range(num_folds - 1):
            self.assertTrue(model_list[i] != model_list[i + 1])
