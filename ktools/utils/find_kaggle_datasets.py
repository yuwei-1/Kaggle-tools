import os
import pandas as pd


def find_competition_info(dir_path : str = "/kaggle/input/"):
    for root, dirs, files in os.walk(dir_path):
        for file in files:
            file_path = os.path.join(root, file)
            if "train.csv" in file_path:
                train_csv_path = file_path
            elif "test.csv" in file_path:
                test_csv_path = file_path
            elif "sample" in file_path:
                sample_sub_csv_path = file_path
    target_col_name = pd.read_csv(train_csv_path).columns[-1]
    return train_csv_path, test_csv_path, sample_sub_csv_path, target_col_name