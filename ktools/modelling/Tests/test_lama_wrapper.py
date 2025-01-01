# import unittest
# import pandas as pd    
# from sklearn.model_selection import KFold
# from ktools.modelling.Automl_models.lama_model import KToolsLAMAWrapper



# class TestKToolsLAMAWrapper(unittest.TestCase):

#     def test_basic_usage(self):
#         train_csv_path = "/Users/yuwei-1/Documents/projects/Kaggle-tools/data/insurance/train.csv"
#         test_csv_path = "/Users/yuwei-1/Documents/projects/Kaggle-tools/data/insurance/test.csv"
#         target_col_name = "Premium Amount"

#         kf = KFold(5, shuffle=True, random_state=42)

#         ktools_lama_wrapper = KToolsLAMAWrapper(train_csv_path,
#                                                 test_csv_path,
#                                                 target_col_name,
#                                                 kf,
#                                                 time_limit=60,
#                                                 save_predictions=True,
#                                                 save_path="./ktools/modelling/Tests/TestData"
#                                                 ).fit()
        
#         oof_pred = ktools_lama_wrapper.predict()