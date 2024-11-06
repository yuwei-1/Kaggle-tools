import unittest
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from sklearn.metrics import mean_squared_error, roc_auc_score, root_mean_squared_error
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from ktools.fitting.cross_validation_executor import CrossValidationExecutor
from ktools.hyperparameter_optimization.model_param_grids import KNNParamGrid
from ktools.modelling.models.hgb_model import HGBModel
from ktools.modelling.models.keras_embedding_model import KerasEmbeddingModel
from ktools.modelling.models.knn_model import KNNModel
from ktools.modelling.models.svm_model import SVMModel
from ktools.utils.data_science_pipeline_settings import DataSciencePipelineSettings


class TestModelSyntax(unittest.TestCase):

    def setUp(self):
        train_csv_path = "/Users/yuwei-1/Documents/projects/Kaggle-tools/data/used_car_prices/train.csv"
        test_csv_path = "/Users/yuwei-1/Documents/projects/Kaggle-tools/data/used_car_prices/test.csv"
        target_col_name = "price"

        settings = DataSciencePipelineSettings(train_csv_path,
                                                test_csv_path,
                                                target_col_name)
        
        train_df, test_df = settings.update()

        self.cat_cols = settings.categorical_col_names
        train_df[settings.categorical_col_names] = train_df[settings.categorical_col_names].astype("category")
        self.X, self.y = train_df.drop(columns=target_col_name), train_df[target_col_name]

    def test_knn(self):
        knn = KNNModel(self.cat_cols, n_neighbors=5, min_max_scaling=True)
        kf = KFold(n_splits=5, shuffle=True, random_state=42)

        def bin_target(tup):
            X,y = tup
            binning_interval = 2000
            y = np.round(y//binning_interval)*binning_interval
            return X, y

        mean_cv_score, oof_predictions, model_list = CrossValidationExecutor(knn,
                                                                            root_mean_squared_error,
                                                                            kf,
                                                                            ).run(self.X, self.y, local_transform_list=[bin_target])
        test_preds = np.zeros(len(self.X[:10]))
        for model in model_list:
            test_preds += model.predict(self.X[:10])/len(model_list)

        print(test_preds)

    def test_hgb(self):
        knn = HGBModel()
        kf = KFold(n_splits=5, shuffle=True, random_state=42)

        mean_cv_score, oof_predictions, model_list = CrossValidationExecutor(knn,
                                                                            root_mean_squared_error,
                                                                            kf,
                                                                            ).run(self.X, self.y)
        test_preds = np.zeros(len(self.X[:10]))
        for model in model_list:
            test_preds += model.predict(self.X[:10])/len(model_list)

        print(test_preds)
        
    def test_gpr(self):
        from sklearn.preprocessing import TargetEncoder        

        class GPR:

            def __init__(self, cat_cols) -> None:
                self.cat_cols = cat_cols
                self.model = GaussianProcessRegressor(ConstantKernel(1.0) * RBF(1.0) + ConstantKernel(1.0) + WhiteKernel(1.0))
                self.target_enc = TargetEncoder(target_type="continuous")
                self.scaler = StandardScaler()

            def fit(self, X, y, **kwargs):
                X[self.cat_cols] = self.target_enc.fit_transform(X[self.cat_cols], y)
                X = self.scaler.fit_transform(X)
                self.model.fit(X, y.to_numpy())
                return self
            
            def predict(self, X):
                X[self.cat_cols] = self.target_enc.transform(X[self.cat_cols])
                X = self.scaler.transform(X)
                y_pred = self.model.predict(X)
                return y_pred

        model = GPR(self.cat_cols)
        kf = KFold(n_splits=5, shuffle=True, random_state=42)

        def subsample(tup):
            X,y = tup
            X,_,y,_ = train_test_split(X, y, random_state=42, test_size=0.8)
            return X, y

        _ = CrossValidationExecutor(model,
                                    root_mean_squared_error,
                                    kf,
                                    ).run(self.X, self.y, local_transform_list=[subsample])
        

    def test_svr(self):
        
        model = SVMModel(self.cat_cols,**{"C" : 1000, "gamma":0.0075, "epsilon":0.1, "max_iter":8000})
        kf = KFold(n_splits=5, shuffle=True, random_state=42)

        def subsample(tup):
            X,y = tup
            X,_,y,_ = train_test_split(X, y, random_state=42, test_size=0.8)
            return X, y
        
        def remove_outliers(train_tuple, threshold=300000):
            X, y = train_tuple
            mask = (y < threshold)
            return X[mask], y[mask]

        _ = CrossValidationExecutor(model,
                                    root_mean_squared_error,
                                    kf,
                                    ).run(self.X, self.y, local_transform_list=[remove_outliers, subsample])
        
    def test_keras_embedding_model(self):
        train_csv_path = "/Users/yuwei-1/Documents/projects/Kaggle-tools/data/loan_approval/train.csv"
        original_csv_path = "/Users/yuwei-1/Documents/projects/Kaggle-tools/data/loan_approval/original.csv"
        test_csv_path = "/Users/yuwei-1/Documents/projects/Kaggle-tools/data/loan_approval/test.csv"
        target_col_name = "loan_status"

        test_data = pd.read_csv(test_csv_path).assign(source=0)
        train_data = pd.read_csv(train_csv_path).assign(source=0)
        original_data = pd.read_csv(original_csv_path).assign(source=1)

        def to_rank(col):
            return col.fillna(-1).rank(method='dense').astype('int') - 1

        def fe(df):
            cat_cols = ['person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file']    
            df['cb_person_cred_hist_length'] = to_rank(df['cb_person_cred_hist_length'])
            df['loan_amnt'] = to_rank(df['loan_amnt'])
            df['person_income'] = to_rank(df['person_income'])
            df['loan_int_rate'] = to_rank(df['loan_int_rate'])
            df['person_emp_length'] = to_rank(df['person_emp_length'])
            df['loan_percent_income'] = to_rank(df['loan_percent_income'])
            df['person_age'] = to_rank(df['person_age'])
            for col in cat_cols:
                col_series = df[col].fillna('#NA#')
                mapping = col_series.value_counts().to_dict()
                code_as = 0
                for i, key in enumerate(reversed(mapping)):
                    mapping[key] = code_as
                    code_as += 1
                df[col] = col_series.map(mapping)
                df[col] = df[col].astype('int')
            return df


        df_all = fe(pd.concat([train_data, test_data, original_data]))

        idxs = (~df_all[target_col_name].isna()) & (df_all['source'] == 0)
        train_data = df_all[idxs].reset_index(drop=True)
        idxs = ( df_all[target_col_name].isna()) & (df_all['source'] == 0)
        test_data = df_all[idxs].drop(columns=[target_col_name])
        original_data = df_all.query('source == 1') 
        train_data.drop(columns="id", inplace=True)
        original_data.drop(columns="id", inplace=True)       

        cont_features = ['cb_person_default_on_file', 'source']
        cat_features = [
            'person_home_ownership',
            'loan_intent',
            'loan_grade',
            'person_emp_length',
            'loan_int_rate',
            'loan_percent_income',
            'person_age',
            'person_income',
            'loan_amnt',
            'cb_person_cred_hist_length']

        cat_features_card = []
        for f in cat_features:
            cat_features_card += [1 + df_all[f].max()]


        features = cat_features + cont_features

        cat_idxs= []
        cont_idxs = []
        for f in cat_features:
            cat_idxs.append([features.index(f)])
        for f in cont_features:
            cont_idxs.append(features.index(f))

        kem = KerasEmbeddingModel(cont_idxs,
                                    cat_idxs,
                                    cat_features_card,
                                    features,
                                    plot_model=True)
        

        target_col_name = "loan_status"

        kf = StratifiedKFold(10, shuffle=True, random_state=42)

        cve = CrossValidationExecutor(kem,
                                    roc_auc_score,
                                    kf,
                                    verbose=2
                                    )

        score_tuple, oofs, model_list = cve.run(train_data.drop(columns=target_col_name),
                                                train_data[[target_col_name]],
                                                additional_data=[original_data.drop(columns=target_col_name), 
                                                                 original_data[[target_col_name]]])
        
        self.assertTrue(score_tuple[0] == 0.9634649329435374)