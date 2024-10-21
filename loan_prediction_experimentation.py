import itertools
import pandas as pd
import numpy as np
import re
import seaborn as sns
import math
import matplotlib.pyplot as plt
from sklearn.cluster import SpectralClustering
from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
import catboost as cb
import optuna
from sklearn.model_selection import StratifiedKFold, cross_val_score
import pprint
from copy import deepcopy
from typing import *
from functools import reduce
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import TargetEncoder
from ktools.fitting.cross_validation_executor import CrossValidationExecutor
from ktools.modelling.models.knn_model import KNNModel
from ktools.preprocessing.basic_feature_transformers import ConvertObjectToCategorical, ConvertToLower, FillNullValues
from ktools.utils.data_science_pipeline_settings import DataSciencePipelineSettings
from sklearn.metrics import root_mean_squared_error, r2_score, roc_auc_score
from sklearn.model_selection import KFold
import xgboost as xgb
from ktools.modelling.models.lgbm_model import LGBMModel
from ktools.modelling.models.xgb_model import XGBoostModel
from ktools.modelling.models.catboost_model import CatBoostModel
from scipy.stats import ks_2samp


if __name__ == "__main__":
    train_csv_path = "/Users/yuwei-1/Documents/projects/Kaggle-tools/data/loan_approval/train.csv"
    test_csv_path = "/Users/yuwei-1/Documents/projects/Kaggle-tools/data/loan_approval/test.csv"
    target_col_name = "loan_status"

    settings = DataSciencePipelineSettings(train_csv_path,
                            test_csv_path,
                            target_col_name)

    
    class LoanApprovalSpecificConverter:
        @staticmethod
        def transform(original_settings : DataSciencePipelineSettings):
            settings = deepcopy(original_settings)
            settings.combined_df['age_discounted_income'] = settings.combined_df['person_income']/settings.combined_df['person_age']
            settings.combined_df['emp_ratio'] = settings.combined_df['person_emp_length']/settings.combined_df['person_age']
            settings.combined_df['cred_hist_ratio'] = settings.combined_df['cb_person_cred_hist_length']/settings.combined_df['person_age']
            settings.combined_df['loan_value_5_years'] = settings.combined_df['loan_amnt'] * (1 + (settings.combined_df['loan_int_rate']/100))**5
            settings.combined_df['payable_in_5_years'] = settings.combined_df['loan_value_5_years'] / (5*settings.combined_df['person_income'])

            settings.training_col_names += [
                                            'age_discounted_income',
                                            'emp_ratio',
                                            'cred_hist_ratio',
                                            'loan_value_5_years',
                                            'payable_in_5_years'
                                            ]
            return settings
        
    class ClassBinner:
        @staticmethod
        def transform(original_settings : DataSciencePipelineSettings):
            settings = deepcopy(original_settings)
            column_name = 'loan_percent_income'
            threshold = 0.1
            settings.combined_df[column_name] = np.round(settings.combined_df[column_name]/threshold)*threshold
            settings.categorical_col_names += [column_name]
            return settings
        
    class AddOGData:
        @staticmethod
        def transform(original_settings : DataSciencePipelineSettings):
            settings = deepcopy(original_settings)
            original_data = pd.read_csv("/Users/yuwei-1/Documents/projects/Kaggle-tools/data/loan_approval/original.csv")

            threshold = 1000
            settings.combined_df['loan_amnt'] = np.round(settings.combined_df['loan_amnt']/threshold)*threshold
            settings.combined_df['person_income'] = np.round(settings.combined_df['person_income']/threshold)*threshold
            settings.combined_df['loan_percent_income'] = settings.combined_df['loan_amnt']/settings.combined_df['person_income']

            settings.train_df = pd.concat([settings.train_df, original_data]).reset_index(drop=True)
            settings.combined_df = pd.concat([settings.train_df, settings.test_df], keys=['train', 'test'])
            settings.categorical_col_names += ['person_income']
            return settings


    class RemoveBadData:
        @staticmethod
        def transform(original_settings : DataSciencePipelineSettings):
            settings = deepcopy(original_settings)

            # age_buffer_emp = 18
            # age_buffer_credit = 18
            settings.combined_df.loc[settings.combined_df['person_age'] > 100, 'person_age'] = None
            settings.combined_df.loc[settings.combined_df['person_emp_length'] > 100, 'person_emp_length'] = None
            # settings.combined_df.loc[settings.combined_df['person_age'] - age_buffer_emp < settings.combined_df['person_emp_length'], 'person_age'] = None
            # settings.combined_df.loc[settings.combined_df['person_age'] - age_buffer_emp < settings.combined_df['person_emp_length'], 'person_emp_length'] = None
            # settings.combined_df.loc[settings.combined_df['person_age'] - age_buffer_credit < settings.combined_df['cb_person_cred_hist_length'], 'cb_person_cred_hist_length'] = None

            return settings
        
    class FrequencyEncoding:
        @staticmethod
        def transform(original_settings : DataSciencePipelineSettings):
            settings = deepcopy(original_settings)
            # settings.categorical_col_names += ['person_income']
            count_cols = [col + "_count" for col in settings.categorical_col_names]
            for i in range(len(count_cols)):

                freq_dict = settings.combined_df[settings.categorical_col_names[i]].value_counts().to_dict()
                settings.combined_df[count_cols[i]] = settings.combined_df[settings.categorical_col_names[i]].map(freq_dict)

            return settings
        

    class CategoricalCombinations:
        @staticmethod
        def transform(original_settings : DataSciencePipelineSettings):
            settings = deepcopy(original_settings)

            combinations = list(itertools.combinations(settings.categorical_col_names + ['person_income'], 2))
            new_col_names = []

            for (feature_1, feature_2) in combinations:
                new_feature_name = feature_1 + "_" + feature_2
                new_col_names += [new_feature_name]
                settings.categorical_col_names += [new_feature_name]
                settings.combined_df[new_feature_name] = settings.combined_df[feature_1].astype('str') + "_" + settings.combined_df[feature_2].astype('str')

            return settings
        
    class SpectralClusteringFeature:
        @staticmethod
        def transform(original_settings : DataSciencePipelineSettings):
            settings = deepcopy(original_settings)
            X, y = settings.combined_df.loc['train'].drop(columns=settings.target_col_name), settings.combined_df.loc['train', settings.target_col_name]


            model = make_pipeline(
                                TargetEncoder(random_state=0, target_type='binary').set_output(transform='pandas'),
                                StandardScaler().set_output(transform='pandas'),
                                SpectralClustering(n_clusters=2)
                                )
            
            classes = model.fit_predict(X, y)

            print(classes)

    class SVCFeature:
        @staticmethod
        def transform(original_settings : DataSciencePipelineSettings):
            settings = deepcopy(original_settings)
            # X, y = settings.combined_df.loc['train'].drop(columns=settings.target_col_name), settings.combined_df.loc['train', settings.target_col_name]

            settings.combined_df['svc_feature'] = 0
            settings.combined_df.loc['train', 'svc_feature'] = pd.read_csv("/Users/yuwei-1/Documents/projects/Kaggle-tools/ktools/modelling/Data/loan_prediction/svc_undersampling_oof.csv", index_col=0).values
            settings.combined_df.loc['test', 'svc_feature'] = pd.read_csv("/Users/yuwei-1/Documents/projects/Kaggle-tools/ktools/modelling/Data/loan_prediction/svc_undersampling_test.csv", index_col=0).values

            settings.combined_df['svc_feature'] = np.round(settings.combined_df['svc_feature']/0.1)*0.1
            settings.categorical_col_names += ['svc_feature']
            # print(settings.combined_df)
            return settings
        
    class MartynovAndreyFeatures:
        @staticmethod
        def transform(original_settings : DataSciencePipelineSettings):
            settings = deepcopy(original_settings)
            settings.combined_df['loantoincome'] = (settings.combined_df['loan_amnt'] / settings.combined_df['person_income']) - settings.combined_df['loan_percent_income']
            # settings.combined_df['log_person_income'] = np.log(settings.combined_df['person_income'])

            # df = settings.combined_df.loc['train'].groupby(['loan_grade', 'loan_intent', 'person_home_ownership'])[settings.target_col_name].agg(['mean']).reset_index()
            # df['gih'] =  df['loan_grade'] + '-' + df['loan_intent'] + '-' + df['person_home_ownership']
            # settings.combined_df['gih'] =  settings.combined_df['loan_grade'] + '-' + settings.combined_df['loan_intent'] + '-' + settings.combined_df['person_home_ownership']
            # alpha = df[['gih', 'mean']].set_index('gih').to_dict()['mean']
            # settings.combined_df['gih'] = settings.combined_df['gih'].astype('category')
            # settings.combined_df['alpha'] = settings.combined_df['gih'].apply(lambda x:alpha[x] if x in alpha else 0.3333)
            # train = settings.combined_df.loc['train']
            # im, em, k = train['person_income'].max(), train['person_emp_length'].max(), 1
            # a = k * em / im 
            # settings.combined_df['beta'] = a * settings.combined_df['person_income'] + settings.combined_df['person_emp_length']
            # test_b['beta'] = a * test_b['income'] + test_b['emp_length']
            # settings.categorical_col_names += ['gih']
            
            return settings
        

    class ConvertEverythingToCategorical():
        @staticmethod
        def transform(settings : DataSciencePipelineSettings):
            cat_cols = settings.training_col_names
            settings.combined_df[cat_cols] = settings.combined_df[cat_cols].astype('category')
            return settings



        
    og_training_col_names = settings.training_col_names

    transforms = [
                # AddOGData.transform,
                # RemoveBadData.transform,
                # ConvertToLower.transform,
                FillNullValues.transform,
                # CategoricalCombinations.transform,
                # ClassBinner.transform,
                # SVCFeature.transform,
                # MartynovAndreyFeatures.transform,
                ConvertEverythingToCategorical.transform,
                ]

    settings = reduce(lambda acc, func: func(acc), transforms, settings)
    settings.update()
    train_df = settings.train_df

    changes = ['convert_all_to_categorical']
    
    # changes = [x[0] + "_" + x[1] for x in changes]
    X, y = train_df.drop(columns=target_col_name), train_df[target_col_name]

    lgb_params = {'objective': 'binary',
                   'metric': 'binary_logloss'}
    
    for change in changes:
        all_scores = []
        for model_random_state in range(10, 15):
            for cv_random_state in range(10, 15):
                model = LGBMModel(random_state = model_random_state, colsample_bytree=0.9, subsample=0.9, **lgb_params)
                kf = StratifiedKFold(5, shuffle=True, random_state=cv_random_state)
                mean_cv_score,_,_ = CrossValidationExecutor(model,
                                                            roc_auc_score,
                                                            kf,
                                                            ).run(X[og_training_col_names], y)
                all_scores += [mean_cv_score[1]]

        # population = np.array(all_scores)
        # np.save("population_of_baselgbm_loan.npy", population)

        original_sample = np.load("population_of_baselgbm_loan.npy")
        new_subsample = np.array(all_scores)

        res = ks_2samp(original_sample, new_subsample)

        significance = 0.05
        print("#"*100)
        print("RESULT: ", res)
        print(f"Original mean: {original_sample.mean()}, New mean: {new_subsample.mean()}")
        print("CHANGE IS USEFUL: ", (res.pvalue < significance) & (original_sample.mean() < new_subsample.mean()))
        print("#"*100)

        history =  pd.read_csv("feature_importance_loan.csv")
        new_df = pd.DataFrame({"change" : [change], "cv_score" : [new_subsample.mean()], "significance" : [res.pvalue]})
        pd.concat([history, new_df]).to_csv("feature_importance_loan.csv", index=False)