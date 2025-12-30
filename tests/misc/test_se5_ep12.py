# from pathlib import Path
# from typing import Tuple
# import numpy as np
# import pandas as pd
# import pytest
# from sklearn.metrics import roc_auc_score
# from sklearn.model_selection import StratifiedKFold

# from ktools.config.dataset import DatasetConfig
# from ktools.models.gbdt.xgboost import XGBoostModel
# from ktools.preprocessing.categorical import CategoricalFrequencyEncoder, CategoricalTargetEncoder
# from ktools.preprocessing.pipe import PreprocessingPipeline


# DATA_PATH = Path("./data/diabetes_prediction/")
# EXPECTED_PATH = Path("./data/diabetes_prediction/expected/")
# TARGET = "diagnosed_diabetes"

# # id split
# SPLIT_ID = 678260
# reset_index = lambda df: df.reset_index(drop=True)

# XGB_PARAMS = {
#     'tree_method': 'hist',
#     'objective': 'binary:logistic',
#     'random_state': 42,
#     'eval_metric': 'auc',
#     'booster': 'gbtree',
#     'n_jobs': -1,
#     'learning_rate': 0.01,
#     "device": "cpu",
#     'lambda': 1.134330929497114,
#     'alpha': 6.780537184218281,
#     'colsample_bytree': 0.1007615968427798,
#     'subsample': 0.7215917619002097,
#     'max_depth': 4,
#     'min_child_weight': 5,
#     'num_boost_round' : 100_000,
#     'early_stopping_rounds' : 200,
# }


# @pytest.fixture
# def prep_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

#     original_data = pd.read_csv(DATA_PATH / "original.csv")
#     training_data = pd.read_csv(DATA_PATH / "train.csv", index_col=0)
#     test_data = pd.read_csv(DATA_PATH / "test.csv", index_col=0).assign(data=0)

#     original_data = original_data.assign(data=2)
#     training_data = training_data.assign(data=0)
#     training_data.iloc[:SPLIT_ID, training_data.columns.get_loc('data')] = 1

#     train_orig_test = pd.concat([training_data, original_data[training_data.columns], test_data], axis=0, ignore_index=True)

#     training_cols = training_data.columns.drop(["data", TARGET]).tolist()

#     orig_target_mean = original_data[TARGET].mean()
#     for c in training_cols:

#         for aggr in ["mean", "count"]:

#             col_name = f'{c}_org_{aggr}'
#             tmp = (original_data.groupby(c)[TARGET]
#                    .agg(aggr)
#                    .rename(col_name)
#                    .reset_index())

#             train_orig_test = train_orig_test.merge(tmp, on=c, how='left')
#             # print(f"Num nan values: ", train_orig_test[col_name].isna().sum())
#             fill_val = orig_target_mean if aggr == 'mean' else 0
#             train_orig_test[col_name] = train_orig_test[col_name].fillna(fill_val)

#     len_train = training_data.shape[0]
#     len_orig = original_data.shape[0]

#     train_data = train_orig_test.iloc[:len_train, :].copy()
#     orig_data = train_orig_test.iloc[len_train:len_train+len_orig, :].copy()
#     test_data = train_orig_test.iloc[len_train+len_orig:, :].copy().drop(columns=[TARGET])

#     return train_data, orig_data, test_data

# @pytest.mark.skip(reason="Long test, used for verification only")
# def test_prep_data(prep_data: Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]):

#     EXPECTED_NORMAL_TRAIN = pd.read_csv(EXPECTED_PATH / "expected_normal_training.csv", index_col=0)
#     EXPECTED_NORMAL_ORIG = pd.read_csv(EXPECTED_PATH / "expected_original.csv", index_col=0)
#     EXPECTED_NORMAL_TEST = pd.read_csv(EXPECTED_PATH / "expected_test_df.csv", index_col=0)

#     train_data, orig_data, test_data = prep_data

#     assert train_data.shape == (700000, 74)
#     assert orig_data.shape == (100000, 74)
#     assert test_data.shape == (300000, 73)

#     pd.testing.assert_frame_equal(reset_index(train_data), reset_index(EXPECTED_NORMAL_TRAIN), check_like=True, check_dtype=False)
#     pd.testing.assert_frame_equal(reset_index(orig_data), reset_index(EXPECTED_NORMAL_ORIG), check_like=True, check_dtype=False)
#     pd.testing.assert_frame_equal(reset_index(test_data), reset_index(EXPECTED_NORMAL_TEST), check_like=True, check_dtype=False)


# @pytest.mark.skip(reason="Long test, used for verification only")
# def test_first_fold_preprocessing_and_training(prep_data: Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]):

#     EXPECTED_FIRST_FOLD_TRAIN = pd.read_csv(EXPECTED_PATH / "fold_0_train_features.csv", index_col=0)
#     EXPECTED_FIRST_FOLD_VAL = pd.read_csv(EXPECTED_PATH / "fold_0_val_features.csv", index_col=0)
#     EXPECTED_FIRST_FOLD_TEST = pd.read_csv(EXPECTED_PATH / "fold_0_test_features.csv", index_col=0)
#     EXPECTED_FIRST_FOLD_TRAIN_WEIGHTS = pd.read_csv(EXPECTED_PATH / "train_weights_0.csv", index_col=0)
#     EXPECTED_FIRST_FOLD_VAL_WEIGHTS = pd.read_csv(EXPECTED_PATH / "val_weights_0.csv", index_col=0)

#     train_data, orig_data, test_data = prep_data


#     all_features = train_data.columns.drop(TARGET).tolist()
#     categorical_features = train_data.drop(columns=TARGET).select_dtypes(include=['object', 'bool']).columns.tolist() + ['family_history_diabetes', 'hypertension_history', 'cardiovascular_history']
#     numerical_features = [col for col in all_features if col not in categorical_features]

#     config = DatasetConfig(
#         training_col_names=all_features,
#         categorical_col_names=categorical_features,
#         numerical_col_names=numerical_features,
#         target_col_name=TARGET,
#     )

#     all_data = pd.concat([train_data, orig_data, test_data], keys=['train', 'orig', 'test'], axis=0)
#     all_data[categorical_features] = all_data[categorical_features].astype('category')
#     train_data, orig_data, test_data = [all_data.xs(key) for key in ['train', 'orig', 'test']]
#     test_data = test_data.drop(columns=[TARGET])

#     kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

#     first_fold = list(kfold.split(train_data, train_data[TARGET]))[0]

#     train_idx, val_idx = first_fold

#     train_fold = train_data.iloc[train_idx]
#     val_fold = train_data.iloc[val_idx]

#     train_fold = pd.concat([train_fold, orig_data], axis=0, ignore_index=True)

#     prep_pipeline = PreprocessingPipeline(preprocessors=[
#         CategoricalFrequencyEncoder(config=config),
#         CategoricalTargetEncoder(config=config),
#     ])

#     train_fold_prepped = prep_pipeline.train_pipe(train_fold)
#     val_fold_prepped = prep_pipeline.inference_pipe(val_fold)
#     test_data_prepped = prep_pipeline.inference_pipe(test_data)

#     train_weights = train_fold_prepped["data"].map({2 : 8, 0 : 16, 1 : 1})
#     val_weights = val_fold_prepped["data"].map({2 : 8, 0 : 16, 1 : 1})


#     pd.testing.assert_frame_equal(
#         reset_index(train_fold_prepped.drop(columns=TARGET)),
#         reset_index(EXPECTED_FIRST_FOLD_TRAIN),
#         check_like=True,
#         check_dtype=False,
#     )

#     pd.testing.assert_frame_equal(
#         reset_index(val_fold_prepped.drop(columns=TARGET)),
#         reset_index(EXPECTED_FIRST_FOLD_VAL),
#         check_like=True,
#         check_dtype=False,
#     )

#     pd.testing.assert_frame_equal(
#         reset_index(test_data_prepped),
#         reset_index(EXPECTED_FIRST_FOLD_TEST),
#         check_like=True,
#         check_dtype=False,
#     )

#     pd.testing.assert_frame_equal(
#         reset_index(train_weights.to_frame()),
#         reset_index(EXPECTED_FIRST_FOLD_TRAIN_WEIGHTS),
#         check_like=True,
#         check_dtype=False,
#     )

#     pd.testing.assert_frame_equal(
#         reset_index(val_weights.to_frame()),
#         reset_index(EXPECTED_FIRST_FOLD_VAL_WEIGHTS),
#         check_like=True,
#         check_dtype=False,
#     )


#     model = XGBoostModel(**XGB_PARAMS)

#     model.fit(
#         train_fold_prepped.drop(columns=[TARGET]),
#         train_fold_prepped[TARGET],
#         validation_set=(val_fold_prepped.drop(columns=[TARGET]), val_fold_prepped[TARGET]),
#         weights=train_weights,
#         val_weights=val_weights,
#     )

#     y_preds = model.predict(val_fold_prepped.drop(columns=[TARGET]))

#     mask = val_fold_prepped["data"] == 0
#     first_fold_score = roc_auc_score(val_fold_prepped.loc[mask, TARGET], y_preds[mask])


#     print(f"First fold AUC: {first_fold_score}")

#     assert np.abs(first_fold_score - 0.7169703) < 1e-6, f"Expected AUC around 0.7169703 but got {first_fold_score}"
