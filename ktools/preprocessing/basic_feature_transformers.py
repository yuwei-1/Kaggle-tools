import numpy as np
from copy import deepcopy
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, MinMaxScaler
from ktools.preprocessing.i_feature_transformer import IFeatureTransformer
from ktools.utils.data_science_pipeline_settings import DataSciencePipelineSettings



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
    

class ConvertObjectToCategorical(IFeatureTransformer):
    @staticmethod
    def transform(original_settings : DataSciencePipelineSettings):
        settings = deepcopy(original_settings)
        cat_cols = settings.categorical_col_names
        settings.combined_df[cat_cols] = settings.combined_df[cat_cols].astype('category')
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
    

class OrdinalEncode(IFeatureTransformer):
    @staticmethod
    def transform(original_settings : DataSciencePipelineSettings):
        settings = deepcopy(original_settings)
        train_df, test_df = settings.update()
        ordinal_encoder = OrdinalEncoder(encoded_missing_value=-1, handle_unknown="use_encoded_value", unknown_value=-1)
        train_df[settings.categorical_col_names] = ordinal_encoder.fit_transform(train_df[settings.categorical_col_names])
        test_df[settings.categorical_col_names] = ordinal_encoder.transform(test_df[settings.categorical_col_names])
        settings.combined_df = pd.concat([train_df, test_df], keys=['train', 'test'])
        return settings

class StandardScaleNumerical(IFeatureTransformer):
    @staticmethod
    def transform(original_settings : DataSciencePipelineSettings):
        settings = deepcopy(original_settings)
        scaler = StandardScaler()
        train_df, test_df = settings.update()
        num_cols = settings.combined_df.select_dtype(include=['number']).columns
        train_df[num_cols] = scaler.fit_transform(train_df[num_cols])
        test_df[num_cols] = scaler.transform(test_df[num_cols])
        settings.combined_df = pd.concat([train_df, test_df], keys=['train', 'test'])
        return settings

class MinMaxScalerNumerical():
    @staticmethod
    def transform(original_settings : DataSciencePipelineSettings):
        settings = deepcopy(original_settings)
        scaler = MinMaxScaler()
        train_df, test_df = settings.update()
        num_cols = settings.combined_df.select_dtypes(include=['number']).columns
        train_df[num_cols] = scaler.fit_transform(train_df[num_cols])
        test_df[num_cols] = scaler.transform(test_df[num_cols])
        settings.combined_df = pd.concat([train_df, test_df], keys=['train', 'test'])
        return settings