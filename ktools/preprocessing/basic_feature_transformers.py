import numpy as np
from copy import deepcopy
import pandas as pd
from ktools.preprocessing.i_feature_transformer import IFeatureTransformer
from ktools.utils.data_science_pipeline_settings import DataSciencePipelineSettings



class ConvertToLower(IFeatureTransformer):
    @staticmethod
    def transform(settings : DataSciencePipelineSettings):
        for col_name in settings.categorical_col_names:
            settings.combined_df[col_name] = settings.combined_df[col_name].str.lower()
        return settings
    

class FillNullValues(IFeatureTransformer):
    @staticmethod
    def transform(settings : DataSciencePipelineSettings, numeric_fill=-1, category_fill='missing'):
        for col_name in settings.training_col_names:
            if pd.api.types.is_numeric_dtype(settings.combined_df[col_name]):
                settings.combined_df[col_name] = settings.combined_df[col_name].fillna(numeric_fill)
            else:
                settings.combined_df[col_name] = settings.combined_df[col_name].fillna(category_fill)
        return settings
    

class ConvertObjectToCategorical(IFeatureTransformer):
    @staticmethod
    def transform(settings : DataSciencePipelineSettings):
        cat_cols = settings.categorical_col_names
        settings.combined_df[cat_cols] = settings.combined_df[cat_cols].astype('category')
        return settings
    

class LogTransformTarget(IFeatureTransformer):
    @staticmethod
    def transform(original_settings : DataSciencePipelineSettings):
        settings = deepcopy(original_settings)
        target = settings.target_col_name
        settings.combined_df['log_' + target] = np.log(settings.combined_df[target] + 1)
        settings.target_col_name = 'log_' + target
        settings.logged = True
        return settings