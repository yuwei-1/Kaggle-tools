import numpy as np
from copy import deepcopy
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, MinMaxScaler, LabelEncoder
from ktools.preprocessing.i_feature_transformer import IFeatureTransformer
from ktools.utils.data_science_pipeline_settings import DataSciencePipelineSettings
from ktools.utils.reduce_dataframe_usage import reduce_dataframe_size


class ConvertToLower(IFeatureTransformer):
    @staticmethod
    def transform(original_settings : DataSciencePipelineSettings):
        settings = deepcopy(original_settings)
        for col_name in settings.categorical_col_names:
            settings.combined_df[col_name] = settings.combined_df[col_name].str.lower()
        return settings
    

class FillNullValues(IFeatureTransformer):
    @staticmethod
    def transform(original_settings : DataSciencePipelineSettings, numeric_fill=-1, category_fill='missing'):
        settings = deepcopy(original_settings)
        for col_name in settings.training_col_names:
            if pd.api.types.is_numeric_dtype(settings.combined_df[col_name]):
                settings.combined_df[col_name] = settings.combined_df[col_name].fillna(numeric_fill)
            else:
                settings.combined_df[col_name] = settings.combined_df[col_name].fillna(category_fill)
        return settings
    
class FillInfValues(IFeatureTransformer):
    @staticmethod
    def transform(original_settings : DataSciencePipelineSettings, pos_inf_fill : float = -2, neg_inf_fill : float = -3):
        settings = deepcopy(original_settings)
        for col_name in settings.numerical_col_names:
            settings.combined_df.loc[settings.combined_df[col_name] == np.inf, col_name] = pos_inf_fill
            settings.combined_df.loc[settings.combined_df[col_name] == -np.inf, col_name] = neg_inf_fill
        return settings
    

class FillNullMeanValues(IFeatureTransformer):
    @staticmethod
    def transform(original_settings : DataSciencePipelineSettings, category_fill='missing'):
        settings = deepcopy(original_settings)
        for col_name in settings.training_col_names:
            if pd.api.types.is_numeric_dtype(settings.combined_df[col_name]):
                settings.combined_df[col_name] = settings.combined_df[col_name].fillna(settings.combined_df[col_name].mean())
            else:
                settings.combined_df[col_name] = settings.combined_df[col_name].fillna(category_fill)
        return settings

class ImputeNumericalAddIndicator(IFeatureTransformer):
    @staticmethod
    def transform(original_settings : DataSciencePipelineSettings, imputation_strategy='mean', add_indicator=True):
        settings = deepcopy(original_settings)
        train_df, test_df = settings.update()
        added_cols = []
        for col_name in settings.numerical_col_names:
            imputer = SimpleImputer(strategy=imputation_strategy, add_indicator=add_indicator)
            train_transformed = imputer.fit_transform(train_df[[col_name]])
            test_transformed = imputer.transform(test_df[[col_name]])
            if train_transformed.shape[1] > 1:
                nan_col_name = col_name + "_nan"
                settings.combined_df.loc['train', [col_name, nan_col_name]] = train_transformed
                settings.combined_df.loc['test', [col_name, nan_col_name]] = test_transformed
                settings.training_col_names += [nan_col_name]
                added_cols += [nan_col_name]
            else:
                settings.combined_df.loc['train', col_name] = train_transformed
                settings.combined_df.loc['test', col_name] = test_transformed
        settings.numerical_col_names.extend(added_cols)
        return settings

class ConvertObjectToCategorical(IFeatureTransformer):
    @staticmethod
    def transform(original_settings : DataSciencePipelineSettings):
        settings = deepcopy(original_settings)
        cat_cols = settings.categorical_col_names
        settings.combined_df[cat_cols] = settings.combined_df[cat_cols].astype('category')
        return settings
    

class ConvertObjectToStrCategorical(IFeatureTransformer):
    @staticmethod
    def transform(original_settings : DataSciencePipelineSettings):
        settings = deepcopy(original_settings)
        cat_cols = settings.categorical_col_names
        settings.combined_df[cat_cols] = settings.combined_df[cat_cols].astype(str).astype('category')
        return settings


class ConvertAllToCategorical(IFeatureTransformer):
    @staticmethod
    def transform(original_settings : DataSciencePipelineSettings):
        settings = deepcopy(original_settings)
        all_cols = settings.training_col_names
        settings.combined_df[all_cols] = settings.combined_df[all_cols].astype(str).astype('category')
        return settings
    

class LogTransformTarget(IFeatureTransformer):
    @staticmethod
    def transform(original_settings : DataSciencePipelineSettings):
        settings = deepcopy(original_settings)
        target = settings.target_col_name
        settings.combined_df[target] = np.log1p(settings.combined_df[target])
        return settings
    
class NanUnknownCategoricals(IFeatureTransformer):
    @staticmethod
    def transform(original_settings : DataSciencePipelineSettings):
        settings = deepcopy(original_settings)
        train_df, test_df = settings.update()
        for col in settings.categorical_col_names:
            mask = ~test_df[col].isin(train_df[col].unique())
            test_df.loc[mask, col] = np.nan
        settings.combined_df = pd.concat([train_df, test_df], keys=['train', 'test'])
        return settings
    

class OrdinalEncode(IFeatureTransformer):
    @staticmethod
    def transform(original_settings : DataSciencePipelineSettings):
        settings = deepcopy(original_settings)
        train_df, test_df = settings.update()
        ordinal_encoder = OrdinalEncoder(encoded_missing_value=-1, handle_unknown="use_encoded_value", unknown_value=-1)
        train_df[settings.categorical_col_names] = ordinal_encoder.fit_transform(train_df[settings.categorical_col_names])
        test_df[settings.categorical_col_names] = ordinal_encoder.transform(test_df[settings.categorical_col_names])
        settings.combined_df = pd.concat([train_df, test_df], keys=['train', 'test'])
        settings.combined_df[settings.categorical_col_names] = settings.combined_df[settings.categorical_col_names].astype(int)
        return settings
    
class LabelEncode(IFeatureTransformer):
    @staticmethod
    def transform(original_settings : DataSciencePipelineSettings):
        settings = deepcopy(original_settings)
        train_df, test_df = settings.update()
        for col in settings.categorical_col_names:
            label_encoder = LabelEncoder()
            train_df[col] = label_encoder.fit_transform(train_df[col])
            test_df[col] = label_encoder.transform(test_df[col])
        settings.combined_df = pd.concat([train_df, test_df], keys=['train', 'test'])
        settings.combined_df[settings.categorical_col_names] = settings.combined_df[settings.categorical_col_names].astype(int)
        return settings


class StandardScaleNumerical(IFeatureTransformer):
    @staticmethod
    def transform(original_settings : DataSciencePipelineSettings):
        settings = deepcopy(original_settings)
        scaler = StandardScaler()
        train_df, test_df = settings.update()
        # num_cols = settings.combined_df[settings.training_col_names].select_dtypes(include=['number']).columns
        num_cols = settings.numerical_col_names
        train_df[num_cols] = scaler.fit_transform(train_df[num_cols])
        test_df[num_cols] = scaler.transform(test_df[num_cols])
        settings.combined_df = pd.concat([train_df, test_df], keys=['train', 'test'])
        return settings

class MinMaxScalerNumerical(IFeatureTransformer):
    @staticmethod
    def transform(original_settings : DataSciencePipelineSettings):
        settings = deepcopy(original_settings)
        scaler = MinMaxScaler()
        train_df, test_df = settings.update()
        num_cols = settings.combined_df[settings.training_col_names].select_dtypes(include=['number']).columns
        train_df[num_cols] = scaler.fit_transform(train_df[num_cols])
        test_df[num_cols] = scaler.transform(test_df[num_cols])
        settings.combined_df = pd.concat([train_df, test_df], keys=['train', 'test'])
        return settings
    
class ReduceMemoryUsage(IFeatureTransformer):
    @staticmethod
    def transform(original_settings : DataSciencePipelineSettings):
        settings = deepcopy(original_settings)
        settings.combined_df = reduce_dataframe_size(settings.combined_df)
        return settings
    
class RemoveTrainTargetOutliers(IFeatureTransformer):
    @staticmethod
    def transform(original_settings : DataSciencePipelineSettings,
                  lq : float = 0.05,
                  uq : float = 0.95):
        settings = deepcopy(original_settings)
        train_df, test_df = settings.update()

        train_labels = train_df[settings.target_col_name]
        lqr = train_labels.quantile(lq)
        uqr = train_labels.quantile(uq)

        no_outlier_train = train_df.loc[(train_labels > lqr) & (train_labels < uqr)]
        settings.combined_df = pd.concat([no_outlier_train, test_df], keys=['train', 'test'])
        return settings