import itertools
import pandas as pd
import numpy as np
import re
import seaborn as sns
import math
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, log_loss
from sklearn.model_selection import train_test_split, GridSearchCV
import catboost as cb
import optuna
from sklearn.model_selection import StratifiedKFold, cross_val_score
import pprint
from copy import deepcopy
from typing import *
from lightgbm import LGBMClassifier, LGBMRegressor
from catboost import CatBoostClassifier
from functools import reduce
from sklearn.preprocessing import TargetEncoder
from ktools.fitting.cross_validation_executor import CrossValidationExecutor
from ktools.modelling.models.knn_model import KNNModel
from ktools.preprocessing.basic_feature_transformers import ConvertObjectToCategorical, ConvertToLower, FillNullValues
from ktools.preprocessing.categorical_denoiser_prepreprocesser import CategoricalDenoiserPreprocessor
from ktools.metrics.fast_matthew_correlation_coefficient import fast_matthews_corr_coeff
from ktools.preprocessing.categorical_string_label_error_imputator import CategoricalLabelErrorImputator
from ktools.preprocessing.categorical_features_embedder import SortMainCategories
from ktools.preprocessing.kaggle_dataset_manager import KaggleDatasetManager
from ktools.utils.data_science_pipeline_settings import DataSciencePipelineSettings
from sklearn.linear_model import LogisticRegression
from ktools.fitting.cross_validate_then_test_sklearn_model import CrossValidateTestSklearnModel
from sklearn.metrics import root_mean_squared_error, r2_score
from sklearn.model_selection import KFold

import xgboost as xgb
from ktools.modelling.models.lgbm_model import LGBMModel
from ktools.modelling.models.xgb_model import XGBoostModel
from ktools.modelling.models.catboost_model import CatBoostModel
from scipy.stats import ks_2samp


if __name__ == "__main__":

    train_csv_path = "data/used_car_prices/train.csv"
    test_csv_path = "data/used_car_prices/test.csv"
    target_col_name = "price"

    settings = DataSciencePipelineSettings(train_csv_path,
                                            test_csv_path,
                                            target_col_name)

    # settings = ConvertObjectToCategorical.transform(settings)
    # settings.update()
    # train_df = settings.train_df
    # X, y = train_df.drop(columns='price'), train_df['price']

    # all_scores = []
    # for model_random_state in range(10):
    #     for cv_random_state in range(10):
    #         model = LGBMModel(n_jobs = 1, random_state = model_random_state, colsample_bytree=0.9, subsample=0.9)
    #         kf = KFold(5, shuffle=True, random_state=cv_random_state)
    #         mean_cv_score = CrossValidationExecutor(model,
    #                                                 root_mean_squared_error,
    #                                                 kf,
    #                                                 ).run(X, y)
    #         all_scores += [mean_cv_score]

    # print(f"random state variation: {np.mean(all_scores)} +- {np.std(all_scores)}")
    # np.save("population_of_baselgbm", np.array(all_scores))

    class UsedCarSpecificConverter:
        @staticmethod
        def transform(original_settings : DataSciencePipelineSettings):
            settings = deepcopy(original_settings)
            def find_pattern(pattern, text):
                match = re.search(pattern, text)
                if match:
                    return match.group(1)
                else:
                    return None
                
            def transmission(text):
                if 'a/t' in text or 'at' in text or 'automatic' in text:
                    return 'automatic transmission'
                elif 'm/t' in text or 'mt' in text or 'manual' in text:
                    return 'manual transmission'
                elif 'cvt' in text:
                    return 'continuously variable transmission'
                else:
                    return 'other'

            def camshafts(text):
                if 'dohc' in text:
                    # double overhead camshaft
                    return 'dohc'
                elif 'sohc' in text:
                    #single overhead camshaft
                    return 'sohc'
                elif 'ohv' in text:
                    # overhead valve
                    return 'ohv'
                elif 'vtec' in text:
                    # variable valve timing and lift electronic control
                    return 'vtec'
                else:
                    return 'other'

            def injection(text):
                if 'ddi' in text:
                    #direct diesel injection
                    return 'ddi'
                elif 'gdi' in text:
                    #gasoline direct injection
                    return 'gdi'
                elif 'mpfi' in text:
                    # multi-point fuel injection
                    return 'mpfi'
                elif 'pdi' in text:
                    # port fuel injection
                    return 'pdi'
                elif 'tfsi' in text or 'tsi' in text:
                    # turbo stratified injection
                    return 'tfsi'
                elif 'gtdi' in text:
                    # gasoline turbocharged direct injection
                    return 'gtdi'
                elif 'sidi' in text:
                    # spark ignition direct injection
                    return 'sidi'
                else:
                    return 'other'
                

            SERIES_PATTERN = re.compile(r'^[A-Za-z0-9\-]+')
            VERSION_PATTERN = re.compile(r'([0-9]+\.[0-9]+[A-Za-z]*)|([A-Z]+[0-9]*)')
            TRIM_PATTERN = re.compile(r'\b(Base|Sport|Premium|Ultimate|XLT|LZ|LT|Plus|Touring|SE|LE|Limited|Platinum|Performance|S|V6|GT|EX|SX|XLE|SR|SL|SV|XSE|TRD|RS|GranSport|Signature|Quad Cab|DRW|Cabriolet|Carbon Edition|Trail Boss|Prestige|Essence|Reserve|xDrive|4MATIC|PreRunner|EcoBoost|Scat Pack|Competition|Adventure Package|Laramie|Grand Touring|Long Range)\b'.lower())

            def extract_car_features(df):

                if 'model' not in df.columns:
                    raise ValueError("Input DataFrame must contain a 'model' column.")
                
                obj_default_na = 'missing'
                def extract_features(model):
                    series = SERIES_PATTERN.search(model)
                    version = VERSION_PATTERN.search(model)
                    trim = TRIM_PATTERN.search(model)
                    
                    return {
                        'Series': series.group(0) if series else obj_default_na,
                        'Version': version.group(0) if version else obj_default_na,
                        'Trim': trim.group(0) if trim else obj_default_na
                    }
                extracted_features = df['model'].apply(extract_features).apply(pd.Series)
                df = pd.concat([df, extracted_features], axis=1)
                return df
            
            settings.combined_df = extract_car_features(settings.combined_df)
            pattern = r'(\d*\.?\d+)\s*hp'
            settings.combined_df['horsepower'] = settings.combined_df['engine'].apply(lambda x : find_pattern(pattern, x)).astype('float64')
            pattern = r'(\d*\.?\d+)\s*(l|liter)'
            settings.combined_df['liters'] = settings.combined_df['engine'].apply(lambda x : find_pattern(pattern, x)).astype('float64')
            pattern = r'(\d*\.?\d+)\s*cylinder'
            settings.combined_df['cylinders'] = settings.combined_df['engine'].apply(lambda x : find_pattern(pattern, x)).astype('float64')
            pattern = r'(\d*\.?\d+)\s*(-speed|speed)'
            settings.combined_df['speed'] = settings.combined_df['transmission'].apply(lambda x : find_pattern(pattern, x)).astype('float64')
            
            settings.combined_df['Power_to_Weight_Ratio'] = settings.combined_df['horsepower'] / settings.combined_df['liters']
            settings.combined_df['injection'] = settings.combined_df['engine'].apply(lambda x : injection(x)).astype('object')
            settings.combined_df['camshaft'] = settings.combined_df['engine'].apply(lambda x : camshafts(x)).astype('object')
            settings.combined_df['transmission_clean'] = settings.combined_df['transmission'].apply(lambda x : transmission(x)).astype('object')
            # settings.combined_df.loc[(settings.combined_df['model'].str.contains('model y|model x|model s|model 3', regex=True)), 'fuel_type'] = 'electric'
            # settings.combined_df.loc[(settings.combined_df['model'].str.contains('electric')), 'fuel_type'] = 'electric'

            luxury_brands =  ['Mercedes-Benz', 'BMW', 'Audi', 'Porsche', 'Land', 
                    'Lexus', 'Jaguar', 'Bentley', 'Maserati', 'Lamborghini', 
                    'Rolls-Royce', 'Ferrari', 'McLaren', 'Aston', 'Maybach']
            luxury_brands = [s.lower() for s in luxury_brands]
            settings.combined_df['Is_Luxury_Brand'] = settings.combined_df['brand'].apply(lambda x: 1 if x in luxury_brands else 0)

            expensive_ext_color = ['blue caelum','dark sapphire','bianco monocerus','c / c',
                                'ice','tempest','beluga black','bianco icarus metallic','blu eleos',
                                'shadow black','nero noctis','sandstone metallic','lizard green','balloon white','onyx',
                                'donington grey metallic','china blue','diamond white','rosso corsa',
                                    'granite','rosso mars metallic',
                                    'carpathian grey','kemora gray metallic','grigio nimbus','dash','bianco isis','python green',
                                    'fountain blue','custom color','vega blue','designo magno matte',
                                    'brands hatch gray metallic',
                                    'rift metallic','gentian blue metallic',
                                    'arancio borealis','blue',
                                    'aventurine green metallic',
                                    'apex blue','daytona gray pearl effect',
                                    'daytona gray pearl effect w/ black roof','matte white',
                                    'carpathian grey premium metallic','blue metallic','santorini black metallic',
                                    'quartzite grey metallic','carrara white metallic','black',
                                    'kinetic blue',
                                    'nero daytona']

            expensive_int_color = ['dark auburn',
                                'hotspur',
                                'cobalt blue',
                                'beluga hide',
                                'linen',
                                'beluga',
                                'black / brown',
                                'nero ade',
                                'sahara tan',
                                'portland']
            
            current_year = 2024

            settings.combined_df['Vehicle_Age'] = current_year - settings.combined_df['model_year']
            settings.combined_df['Mileage_per_Year'] = settings.combined_df['milage'] / (settings.combined_df['Vehicle_Age'] + 1)
            settings.combined_df['milage_with_age'] =  settings.combined_df.groupby('Vehicle_Age')['milage'].transform('mean')
            settings.combined_df['Mileage_per_Year_with_age'] =  settings.combined_df.groupby('Vehicle_Age')['Mileage_per_Year'].transform('mean')

            settings.combined_df['expensive_ext_col'] = settings.combined_df['ext_col'].isin(expensive_ext_color).astype(int)
            settings.combined_df['expensive_int_col'] = settings.combined_df['int_col'].isin(expensive_int_color).astype(int)
            settings.combined_df['twin_turbo'] = settings.combined_df['engine'].str.contains('twin turbo').astype(int)
            settings.combined_df['turbo'] = settings.combined_df['engine'].str.contains('turbo').astype(int)
            settings.combined_df['length_model'] = settings.combined_df['model'].apply(lambda x : len(x))
            settings.combined_df['length_ext_col'] = settings.combined_df['ext_col'].apply(lambda x : len(x))
            settings.combined_df['length_int_col'] = settings.combined_df['int_col'].apply(lambda x : len(x))
            
            # clean_colors = ['ext_col', 'int_col']
            # string_imputator = CategoricalLabelErrorImputator(verbose=True)
            # settings.combined_df[['basic_ext_color', 'basic_int_color']] = string_imputator.impute(settings.combined_df[clean_colors],
            #                                                                                                             clean_colors,
            #                                                                                                             1500)
            
            # settings.combined_df['basic_ext_color'] = settings.combined_df['basic_ext_color'].astype('object')
            # settings.combined_df['basic_int_color'] = settings.combined_df['basic_int_color'].astype('object')
            
            settings.training_col_names += ['horsepower', 
                                            'injection',
                                            'camshaft', 
                                            'cylinders', 
                                            'expensive_ext_col', 
                                            'expensive_int_col', 
                                            'twin_turbo', 
                                            'turbo',
                                            'transmission_clean',
                                            'speed',
                                            # 'basic_ext_color',
                                            # 'basic_int_color',
                                            'liters',
                                            'length_model',
                                            'length_ext_col',
                                            'length_int_col',
                                            'Series',
                                            'Version',
                                            'Trim',
                                            'Is_Luxury_Brand',
                                            'Power_to_Weight_Ratio',
                                            'Vehicle_Age',
                                            'Mileage_per_Year',
                                            'milage_with_age',
                                            'Mileage_per_Year_with_age',
                                            ]
            
            settings.categorical_col_names += ['injection',
                                            'camshaft',
                                            'transmission_clean',
                                            # 'basic_ext_color',
                                            # 'basic_int_color',
                                            'Series',
                                            'Version',
                                            'Trim',
                                            ]
                                            
            
            return settings
        
    class CreateMAEFeature:
        @staticmethod
        def transform(original_settings : DataSciencePipelineSettings):
            settings = deepcopy(original_settings)
            X, y = settings.combined_df.loc['train'].drop(columns=settings.target_col_name), settings.combined_df.loc['train', settings.target_col_name]
            params = {
                    'objective': 'MAE',
                    'random_state': 42,
                    'colsample_bytree' : 0.9,
                    'subsample' : 0.9,
                    'n_jobs' : 1
                }
            
            model = LGBMModel(**params)
            kf = KFold(5, shuffle=True, random_state=42)
            mean_cv_score, mae_oof, _ = CrossValidationExecutor(model,
                                                            root_mean_squared_error,
                                                            kf,
                                                            ).run(X, y)
            settings.combined_df['MAE_feature'] = 0
            settings.combined_df.loc['train', 'MAE_feature'] = mae_oof 
            return settings

    class CreateRMSEFeature:
        @staticmethod
        def transform(original_settings : DataSciencePipelineSettings):
            settings = deepcopy(original_settings)
            X, y = settings.combined_df.loc['train'].drop(columns=settings.target_col_name), settings.combined_df.loc['train', settings.target_col_name]
            params = {
                    'objective': 'RMSE',
                    'random_state': 42,
                    'colsample_bytree' : 0.9,
                    'subsample' : 0.9,
                    'n_jobs' : 1
                }
            
            model = LGBMModel(**params)
            kf = KFold(5, shuffle=True, random_state=42)
            mean_cv_score, mae_oof, _ = CrossValidationExecutor(model,
                                                            root_mean_squared_error,
                                                            kf,
                                                            ).run(X, y)
            settings.combined_df['RMSE_feature'] = 0
            settings.combined_df.loc['train', 'RMSE_feature'] = mae_oof 
            return settings
        
    class CreateTweedieFeature:
        @staticmethod
        def transform(original_settings : DataSciencePipelineSettings):
            settings = deepcopy(original_settings)
            X, y = settings.combined_df.loc['train'].drop(columns=settings.target_col_name), settings.combined_df.loc['train', settings.target_col_name]
            params = {
                    'objective': 'tweedie',
                    'random_state': 42,
                    'colsample_bytree' : 0.9,
                    'subsample' : 0.9,
                    'n_jobs' : 1
                }
            
            model = LGBMModel(**params)
            kf = KFold(5, shuffle=True, random_state=42)
            mean_cv_score, mae_oof, _ = CrossValidationExecutor(model,
                                                            root_mean_squared_error,
                                                            kf,
                                                            ).run(X, y)
            settings.combined_df['tweedie_feature'] = 0
            settings.combined_df.loc['train', 'tweedie_feature'] = mae_oof 
            return settings
        
    class CreateHuberFeature:
        @staticmethod
        def transform(original_settings : DataSciencePipelineSettings):
            settings = deepcopy(original_settings)
            X, y = settings.combined_df.loc['train'].drop(columns=settings.target_col_name), settings.combined_df.loc['train', settings.target_col_name]
            params = {
                    'objective': 'huber',
                    'random_state': 42,
                    'colsample_bytree' : 0.9,
                    'subsample' : 0.9,
                    'n_jobs' : 1
                }
            
            model = LGBMModel(**params)
            kf = KFold(5, shuffle=True, random_state=42)
            mean_cv_score, mae_oof, _ = CrossValidationExecutor(model,
                                                            root_mean_squared_error,
                                                            kf,
                                                            ).run(X, y)
            settings.combined_df['huber_feature'] = 0
            settings.combined_df.loc['train', 'huber_feature'] = mae_oof 
            return settings
        
    class CreateMCEFeature:
        @staticmethod
        def transform(original_settings : DataSciencePipelineSettings):
            settings = deepcopy(original_settings)
            X, y = settings.combined_df.loc['train'].drop(columns=settings.target_col_name), settings.combined_df.loc['train', settings.target_col_name]

            def mce(preds, train_data):
                y_true = train_data.get_label()
                residual = preds - y_true
                grad = residual**3
                hess = 3*residual**2
                return grad, hess
            
            def mce_metric(preds, train_data):
                y_true = train_data.get_label()
                mce = np.mean((preds - y_true)**3)
                return 'mce', mce, False
            
            params = {
                    'random_state': 42,
                    'colsample_bytree' : 0.9,
                    'subsample' : 0.9,
                    'n_jobs' : 1,
                    'objective' : mce,
                    'metric' : 'mse'
                    # 'feval' : mce_metric
                }
            
            
            model = LGBMModel(**params)
            kf = KFold(5, shuffle=True, random_state=42)
            mean_cv_score, mae_oof, _ = CrossValidationExecutor(model,
                                                            root_mean_squared_error,
                                                            kf,
                                                            ).run(X, y)
            settings.combined_df['mce_feature'] = 0
            settings.combined_df.loc['train', 'mce_feature'] = mae_oof 
            return settings
        
    class XGBoostFeature:
        @staticmethod
        def transform(original_settings : DataSciencePipelineSettings):
            settings = deepcopy(original_settings)
            X, y = settings.combined_df.loc['train'].drop(columns=settings.target_col_name), settings.combined_df.loc['train', settings.target_col_name]
            params = {
                    'objective': 'reg:squarederror',
                    'num_boost_round' : 1000,
                    'stopping_rounds' : 50,
                    'random_state': 42,
                    'colsample_bytree' : 0.25,
                    # 'subsample' : 0.9,
                    'n_jobs' : 1
                }
            
            model = XGBoostModel(**params)
            kf = KFold(5, shuffle=True, random_state=42)
            mean_cv_score, mae_oof, _ = CrossValidationExecutor(model,
                                                            root_mean_squared_error,
                                                            kf,
                                                            ).run(X, y)
            settings.combined_df['xgb_feature'] = 0
            settings.combined_df.loc['train', 'xgb_feature'] = mae_oof 
            return settings
        
    class CatBoostFeature:
        @staticmethod
        def transform(original_settings : DataSciencePipelineSettings):
            settings = deepcopy(original_settings)
            X, y = settings.combined_df.loc['train'].drop(columns=settings.target_col_name), settings.combined_df.loc['train', settings.target_col_name]
            params = {
                    # 'objective': 'reg:squarederror',
                    'num_boost_round' : 1000,
                    'stopping_rounds' : 50,
                    'random_state': 42,
                }
            
            model = CatBoostModel(**params)
            kf = KFold(5, shuffle=True, random_state=42)
            mean_cv_score, mae_oof, _ = CrossValidationExecutor(model,
                                                            root_mean_squared_error,
                                                            kf,
                                                            ).run(X, y)
            settings.combined_df['cat_feature'] = 0
            settings.combined_df.loc['train', 'cat_feature'] = mae_oof 
            return settings

    class WithoutOutliersFeature:
        @staticmethod
        def transform(original_settings : DataSciencePipelineSettings):
            settings = deepcopy(original_settings)
            X, y = settings.combined_df.loc['train'].drop(columns=settings.target_col_name), settings.combined_df.loc['train', settings.target_col_name]
            params = {
                    'objective': 'rmse',
                    'random_state': 42,
                    # 'colsample_bytree' : 0.9,
                    # 'subsample' : 0.9,
                    'n_jobs' : 1
                }
            
            def remove_outliers(train_tuple, threshold=300000):
                X, y = train_tuple
                mask = (y < threshold)
                return X[mask], y[mask]
            
            model = LGBMModel(**params)
            kf = KFold(5, shuffle=True, random_state=42)
            mean_cv_score, mae_oof, _ = CrossValidationExecutor(model,
                                                            root_mean_squared_error,
                                                            kf,
                                                            ).run(X, y, local_transform_list=[remove_outliers])
            settings.combined_df['without_outlier_feature'] = 0
            settings.combined_df.loc['train', 'without_outlier_feature'] = mae_oof

            return settings
        
    
    class AllLGBMFeatures:
        @staticmethod
        def transform(original_settings : DataSciencePipelineSettings):
            settings = deepcopy(original_settings)

            # X, y = settings.combined_df.loc['train'].drop(columns=[settings.target_col_name]), settings.combined_df.loc['train', settings.target_col_name]
            # params = {
            #         'objective': 'rmse',
            #         'random_state': 42,
            #         # 'colsample_bytree' : 0.9,
            #         # 'subsample' : 0.9,
            #         'n_jobs' : 1
            #     }
            
            # def remove_outliers(train_tuple, threshold=300000):
            #     X, y = train_tuple
            #     mask = (y < threshold)
            #     return X[mask], y[mask]
            
            # model = LGBMModel(**params)
            # kf = KFold(5, shuffle=True, random_state=42)
            # mean_cv_score, mae_oof, _ = CrossValidationExecutor(model,
            #                                                 root_mean_squared_error,
            #                                                 kf,
            #                                                 ).run(X, y, local_transform_list=[remove_outliers])
            # settings.combined_df['without_outlier_feature'] = 0
            # settings.combined_df.loc['train', 'without_outlier_feature'] = mae_oof

            X, y = settings.combined_df.loc['train'].drop(columns=[settings.target_col_name]), settings.combined_df.loc['train', settings.target_col_name]

            params = {
                    'objective': 'MAE',
                    'n_estimators': 1000,
                    'random_state': 42,
                    # 'colsample_bytree' : 0.9,
                    # 'subsample' : 0.9,
                    'n_jobs' : 1
                }

            model = LGBMModel(**params)
            kf = KFold(5, shuffle=True, random_state=42)
            mean_cv_score, mae_oof, _ = CrossValidationExecutor(model,
                                                            root_mean_squared_error,
                                                            kf,
                                                            ).run(X, y)
            settings.combined_df['MAE_feature'] = 0.0
            settings.combined_df.loc['train', 'MAE_feature'] = mae_oof


            X, y = settings.combined_df.loc['train'].drop(columns=[settings.target_col_name]), settings.combined_df.loc['train', settings.target_col_name]
            params = {
                    'objective': 'rmse',
                    'random_state': 42,
                    # 'colsample_bytree' : 0.9,
                    # 'subsample' : 0.9,
                    'n_jobs' : 1
                }
                        
            model = LGBMModel(**params)
            kf = KFold(5, shuffle=True, random_state=42)
            mean_cv_score, mae_oof, _ = CrossValidationExecutor(model,
                                                            root_mean_squared_error,
                                                            kf,
                                                            ).run(X, y)
            settings.combined_df['rmse_feat'] = 0
            settings.combined_df.loc['train', 'rmse_feat'] = mae_oof

            settings.combined_df['diff'] = 0.0
            settings.combined_df.loc['train', 'diff'] = settings.combined_df.loc['train', 'rmse_feat'] - settings.combined_df.loc['train', 'MAE_feature']

            return settings


    class BrandModelRarity:
        @staticmethod
        def transform(original_settings : DataSciencePipelineSettings):
            settings = deepcopy(original_settings)
            settings.combined_df["brand_model"] = settings.combined_df['brand'].astype('str') + "_" + settings.combined_df['model'].astype('str')
            count_dict = settings.combined_df["brand_model"].value_counts().to_dict()
            settings.combined_df["brand_model_count"] = settings.combined_df["brand_model"].apply(lambda x : count_dict[x])
            return settings


    class ClipOutliersFeature:
        @staticmethod
        def transform(original_settings : DataSciencePipelineSettings):
            settings = deepcopy(original_settings)
            X, y = settings.combined_df.loc['train'].drop(columns=settings.target_col_name), settings.combined_df.loc['train', settings.target_col_name]
            params = {
                    'objective': 'rmse',
                    'random_state': 42,
                    'colsample_bytree' : 0.9,
                    'subsample' : 0.9,
                    'n_jobs' : 1
                }
            
            def clip_outliers(train_tuple, threshold=500000):
                X, y = train_tuple
                mask = (y > threshold)
                # X.loc[mask] = threshold
                y.loc[mask] = threshold
                return X, y
            
            model = LGBMModel(**params)
            kf = KFold(5, shuffle=True, random_state=42)
            mean_cv_score, mae_oof, _ = CrossValidationExecutor(model,
                                                            root_mean_squared_error,
                                                            kf,
                                                            ).run(X, y, local_transform_list=[clip_outliers])
            settings.combined_df['clipped_outlier_feature'] = 0
            settings.combined_df.loc['train', 'clipped_outlier_feature'] = mae_oof 
            return settings
        

    class DenoiseTargetLGBMFeature:
        @staticmethod
        def transform(original_settings : DataSciencePipelineSettings):
            settings = deepcopy(original_settings)
            X, y = settings.combined_df.loc['train'].drop(columns=settings.target_col_name), settings.combined_df.loc['train', settings.target_col_name]
            
            round_to_nearest = 10000
            y = (y//round_to_nearest)*round_to_nearest
            # classes, _ = round_thousand.factorize()
            # _y = pd.Series(classes, index=X.index)
            # mapping = dict(zip(classes, round_thousand))

            params = {
                    'objective': 'rmse',
                    # 'num_class' : _y.nunique(),
                    # 'metric' : 'multi_logloss',
                    'random_state': 42,
                    'colsample_bytree' : 0.9,
                    'subsample' : 0.9,
                    'n_jobs' : 1
                }
            
            model = LGBMModel(**params)

            # def softmax(logits):
            #     exponentiate = np.exp(logits)
            #     denom = exponentiate.sum(axis=1).reshape(-1,1)
            #     return exponentiate/denom
            
            kf = KFold(5, shuffle=True, random_state=42)
            mean_cv_score, oof, _ = CrossValidationExecutor(model,
                                                            root_mean_squared_error,
                                                            kf).run(X, y)
            
            settings.combined_df['reg_round10000'] = 0
            settings.combined_df.loc['train', 'reg_round10000'] = oof

            # print(root_mean_squared_error(y, settings.combined_df.loc['train', 'mutliclass_round1000'].apply(lambda x : mapping[x])))
            return settings
        

    class StandardizeFeatures:
        @staticmethod
        def transform(original_settings : DataSciencePipelineSettings):
            settings = deepcopy(original_settings)
            numerical_columns = settings.combined_df.select_dtypes(include=['number']).columns.tolist()
            numerical_columns.remove(settings.target_col_name)

            numerical_scaler = StandardScaler()
            settings.combined_df[numerical_columns] = numerical_scaler.fit_transform(settings.combined_df[numerical_columns])
            return settings
        
    class RemoveRareCategories:
        @staticmethod
        def transform(original_settings : DataSciencePipelineSettings):
            settings = deepcopy(original_settings)
            threshold = 500
            for col_name in settings.categorical_col_names:
                vc = settings.combined_df[col_name].value_counts()
                rare_cats = vc.loc[vc<threshold].index.values
                settings.combined_df.loc[settings.combined_df[col_name].isin(rare_cats), col_name] = "rare"
            return settings
        
    class FrequencyEncoding:
        @staticmethod
        def transform(original_settings : DataSciencePipelineSettings):
            settings = deepcopy(original_settings)

            count_cols = [col + "_count" for col in settings.categorical_col_names]
            for i in range(len(count_cols)):

                freq_dict = settings.combined_df[settings.categorical_col_names[i]].value_counts().to_dict()
                settings.combined_df[count_cols[i]] = settings.combined_df[settings.categorical_col_names[i]].map(freq_dict)

            return settings
        
    class KNNFeature:
        @staticmethod
        def transform(original_settings : DataSciencePipelineSettings):
            settings = deepcopy(original_settings)
            X, y = settings.combined_df.loc['train'].drop(columns=settings.target_col_name), settings.combined_df.loc['train', settings.target_col_name]

            # tenc = TargetEncoder(smooth=5, target_type="continuous")
            # X[settings.categorical_col_names] = tenc.fit_transform(X[settings.categorical_col_names], y)

            
            # knn = KNeighborsRegressor()
            knn = KNNModel(settings.categorical_col_names,
                           n_neighbors=1000,
                           )

            kf = KFold(5, shuffle=True, random_state=42)
            mean_cv_score, oof, _ = CrossValidationExecutor(knn,
                                                            root_mean_squared_error,
                                                            kf).run(X, y)
            
            settings.combined_df['knn_feature'] = 0
            settings.combined_df.loc['train', 'knn_feature'] = oof
            return settings
        

    class CategoricalCombinations:
        @staticmethod
        def transform(original_settings : DataSciencePipelineSettings):
            settings = deepcopy(original_settings)

            round_to_nearest = 10000
            settings.combined_df['binned_milage'] = (settings.combined_df['milage']//round_to_nearest)*round_to_nearest
            settings.combined_df['binned_model_year'] = (settings.combined_df['model_year']//5)*5
            combinations = list(itertools.combinations(settings.categorical_col_names + ['binned_milage', 'binned_model_year'], 2))
            new_col_names = []

            for (feature_1, feature_2) in combinations:
                new_feature_name = feature_1 + "_" + feature_2
                new_col_names += [new_feature_name]
                settings.categorical_col_names += [new_feature_name]
                settings.combined_df[new_feature_name] = settings.combined_df[feature_1].astype('str') + "_" + settings.combined_df[feature_2].astype('str')

            return settings
        
    


    og_training_col_names = settings.training_col_names

    transforms = [ConvertToLower.transform,
                #   UsedCarSpecificConverter.transform,
                #   RemoveRareCategories.transform,
                #   FrequencyEncoding.transform,
                #   FillNullValues.transform,
                #   CategoricalCombinations.transform,
                  BrandModelRarity.transform,
                  ConvertObjectToCategorical.transform,
                #   AllLGBMFeatures.transform
                #   DenoiseTargetLGBMFeature.transform,
                #   StandardizeFeatures.transform,
                #   CatBoostFeature.transform
                  ]
    

    settings = reduce(lambda acc, func: func(acc), transforms, settings)
    settings.update()
    train_df = settings.train_df
    X, y = train_df.drop(columns='price'), train_df['price']

    all_scores = []
    for model_random_state in range(5):
        for cv_random_state in range(5):
            model = LGBMModel(n_jobs = 1, random_state = model_random_state, colsample_bytree=0.9, subsample=0.9)
            kf = KFold(5, shuffle=True, random_state=cv_random_state)
            mean_cv_score,_,_ = CrossValidationExecutor(model,
                                                        root_mean_squared_error,
                                                        kf,
                                                        ).run(X[og_training_col_names + ['brand_model_count']], y)
            all_scores += [mean_cv_score[1]]

    original_sample = np.load("population_of_baselgbm.npy")
    new_subsample = np.array(all_scores)
    res = ks_2samp(original_sample, new_subsample)

    significance = 0.05
    print("#"*100)
    print("RESULT: ", res)
    print(f"Original mean: {original_sample.mean()}, New mean: {new_subsample.mean()}")
    print("CHANGE IS USEFUL: ", (res.pvalue < significance) & (original_sample.mean() > new_subsample.mean()))
    print("#"*100)

    history =  pd.read_csv("feature_importance.csv")
    new_df = pd.DataFrame({"change" : ["brand_model_count"], "cv_score" : [new_subsample.mean()], "significance" : [res.pvalue]})
    pd.concat([history, new_df]).to_csv("feature_importance.csv", index=False)


    # change_name = ['brand_model', 'brand_fuel_type', 'brand_engine', 'brand_transmission', 'brand_ext_col', 'brand_int_col', 'brand_accident', 'brand_clean_title', 'brand_binned_milage', 'brand_binned_model_year', 'model_fuel_type', 'model_engine', 'model_transmission', 'model_ext_col', 'model_int_col', 'model_accident', 'model_clean_title', 'model_binned_milage', 'model_binned_model_year', 'fuel_type_engine', 'fuel_type_transmission', 'fuel_type_ext_col', 'fuel_type_int_col', 'fuel_type_accident', 'fuel_type_clean_title', 'fuel_type_binned_milage', 'fuel_type_binned_model_year', 'engine_transmission', 'engine_ext_col', 'engine_int_col', 'engine_accident', 'engine_clean_title', 'engine_binned_milage', 'engine_binned_model_year', 'transmission_ext_col', 'transmission_int_col', 'transmission_accident', 'transmission_clean_title', 'transmission_binned_milage', 'transmission_binned_model_year', 'ext_col_int_col', 'ext_col_accident', 'ext_col_clean_title', 'ext_col_binned_milage', 'ext_col_binned_model_year', 'int_col_accident', 'int_col_clean_title', 'int_col_binned_milage', 'int_col_binned_model_year', 'accident_clean_title', 'accident_binned_milage', 'accident_binned_model_year', 'clean_title_binned_milage', 'clean_title_binned_model_year', 'binned_milage_binned_model_year']
    
    # print(X.head(5))
    
    # cv_scores = []
    # significances = []
    # for name in change_name:

    #     all_scores = []
    #     for model_random_state in range(5):
    #         for cv_random_state in range(5):
    #             model = LGBMModel(n_jobs = 1, random_state = model_random_state, colsample_bytree=0.9, subsample=0.9)
    #             kf = KFold(5, shuffle=True, random_state=cv_random_state)
    #             mean_cv_score, _ = CrossValidationExecutor(model,
    #                                                     root_mean_squared_error,
    #                                                     kf,
    #                                                     ).run(X[og_training_col_names + [name]], y)
    #             all_scores += [mean_cv_score]

    #     original_sample = np.load("population_of_baselgbm.npy")
    #     new_subsample = np.array(all_scores)
    #     res = ks_2samp(original_sample, new_subsample)

    #     significance = 0.05
    #     print("#"*100)
    #     print("RESULT: ", res)
    #     print(f"Original mean: {original_sample.mean()}, New mean: {new_subsample.mean()}")
    #     print("CHANGE IS USEFUL: ", (res.pvalue < significance) & (original_sample.mean() > new_subsample.mean()))
    #     print("#"*100)

    #     cv_scores += [new_subsample.mean()]
    #     significances += [res.pvalue]

    #     history =  pd.read_csv("feature_importance.csv")
    #     new_df = pd.DataFrame({"change" : [name], "cv_score" : [new_subsample.mean()], "significance" : [res.pvalue]})
    #     pd.concat([history, new_df]).to_csv("feature_importance.csv", index=False)