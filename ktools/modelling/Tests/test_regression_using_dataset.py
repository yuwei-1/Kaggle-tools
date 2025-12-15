from functools import reduce
import unittest
from sklearn.metrics import r2_score
from parameterized import parameterized
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from ktools.modelling.Interfaces.i_ktools_model import IKtoolsModel
from ktools.modelling.ktools_models.catboost_model import CatBoostModel
from ktools.modelling.ktools_models.lgbm_model import LGBMModel
from ktools.preprocessing.basic_feature_transformers import *
from ktools.utils.data_science_pipeline_settings import DataSciencePipelineSettings
from ktools.modelling.ktools_models.xgb_model import XGBoostModel


class TestKtoolsRegressionUsingDataset(unittest.TestCase):
    def setUp(self) -> None:
        train_csv_path = (
            "/Users/yuwei-1/Documents/projects/Kaggle-tools/data/insurance/train.csv"
        )
        test_csv_path = (
            "/Users/yuwei-1/Documents/projects/Kaggle-tools/data/insurance/test.csv"
        )
        self.target_col_name = "Premium Amount"

        settings = DataSciencePipelineSettings(
            train_csv_path,
            test_csv_path,
            self.target_col_name,
            train_data=pd.read_csv(train_csv_path)
            .sample(1000, random_state=42)
            .reset_index(drop=True),
        )

        class FE:
            @staticmethod
            def transform(original_settings: DataSciencePipelineSettings):
                settings = deepcopy(original_settings)
                settings.combined_df["Policy Start Date"] = pd.to_datetime(
                    settings.combined_df["Policy Start Date"]
                )
                settings.combined_df["year"] = settings.combined_df[
                    "Policy Start Date"
                ].dt.year
                settings.combined_df["month"] = settings.combined_df[
                    "Policy Start Date"
                ].dt.month
                settings.combined_df["Policy Start Date"] = settings.combined_df[
                    "Policy Start Date"
                ].astype(str)
                settings.combined_df["cat_health_score"] = settings.combined_df[
                    "Health Score"
                ]
                settings.categorical_col_names += ["cat_health_score"]
                return settings

        transforms = [
            FE.transform,
            FillNullValues.transform,
            OrdinalEncode.transform,
            ConvertObjectToCategorical.transform,
        ]
        self.settings = settings = reduce(
            lambda acc, func: func(acc), transforms, settings
        )
        X, y, X_test, _ = settings.get_data()
        self.data_sets = train_test_split(
            X, y, random_state=42, shuffle=True, test_size=0.2
        )

    @parameterized.expand(
        [
            (
                XGBoostModel(
                    **{"objective": "reg:squarederror", "early_stopping_rounds": None}
                ),
                XGBRegressor(enable_categorical=True),
            ),
            (
                CatBoostModel(
                    **{
                        "loss_function": "RMSE",
                        "random_state": None,
                        "early_stopping_rounds": None,
                        "num_boost_round": 1000,
                    }
                ),
                CatBoostRegressor(),
            ),
            (
                LGBMModel(**{"objective": "regression", "early_stopping_rounds": None}),
                LGBMRegressor(),
            ),
        ]
    )
    def test_compare_regression_scores(self, model_cls: IKtoolsModel, original_model):
        if isinstance(original_model, CatBoostRegressor):
            original_model = CatBoostRegressor(
                cat_features=self.settings.categorical_col_names,
                allow_writing_files=False,
            )

        X_train, X_valid, y_train, y_valid = self.data_sets
        model_cls = model_cls.fit(X_train, y_train)
        original_model = original_model.fit(X_train, y_train)

        y_pred_custom = model_cls.predict(X_valid)
        y_pred_expected = original_model.predict(X_valid)

        custom_score = r2_score(y_valid, y_pred_custom)
        expected_score = r2_score(y_valid, y_pred_expected)

        self.assertEqual(custom_score, expected_score)
