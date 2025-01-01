import unittest

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from ktools.experimentation.robust_feature_importance_checker import RobustFeatureImportanceChecker


class TestRobustFeatureImportanceChecker(unittest.TestCase):

    def test_mental_health(self):
        
        class FE:
            def create(self, df):
                df['PS_ratio']=df['Academic Pressure']/df['Job Satisfaction']
                df['PSF_ratio']=df['PS_ratio']*df['Financial Stress']
                df['PF_factor']=df['Academic Pressure']*df['Financial Stress']
                return df, ['PS_ratio','PSF_ratio','PF_factor']
        
        train_csv_path = "/Users/yuwei-1/Documents/projects/Kaggle-tools/data/mental_health/train.csv"
        test_csv_path = "/Users/yuwei-1/Documents/projects/Kaggle-tools/data/mental_health/test.csv"
        target_col_name = "Depression"

        kf = StratifiedKFold(5, shuffle=True, random_state=42)
        checker = RobustFeatureImportanceChecker(train_csv_path,
                                                test_csv_path,
                                                target_col_name,
                                                FE(),
                                                kf,
                                                roc_auc_score,
                                                model_params={'objective': 'binary', 'metric': 'binary_logloss'},
                                                result_path="./ktools/experimentation/Tests/TestData/mental_health"
                                                )
        
        checker.run()