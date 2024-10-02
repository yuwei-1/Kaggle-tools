import sys
import numpy as np
from sklearn.linear_model import Lasso, LinearRegression
sys.path.append("/Users/yuwei-1/Documents/projects/Kaggle-tools")
import pandas as pd
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import KFold, StratifiedKFold
from ktools.fitting.cross_validate_then_test_sklearn_model import CrossValidateTestSklearnModel
from ktools.modelling.models.catboost_model import CatBoostModel
from ktools.modelling.models.lgbm_model import LGBMModel
from ktools.modelling.models.xgb_model import XGBoostModel
from ktools.preprocessing.kaggle_dataset_manager import KaggleDatasetManager


class MetaModelExperiment:

    def __init__(self) -> None:
        pass

    def run_used_car_experiment(self):

        train_df = pd.read_csv("/Users/yuwei-1/Documents/projects/Kaggle-tools/data/used_car_prices/basic_train.csv", index_col=0)
        test_df = pd.read_csv("/Users/yuwei-1/Documents/projects/Kaggle-tools/data/used_car_prices/basic_test.csv", index_col=0)

        # y = train_df.pop('price')

        cat_cols = [col_name for col_name in train_df.columns if train_df[col_name].dtype == 'object']

        train_df[cat_cols] = train_df[cat_cols].astype('category')
        test_df[cat_cols] = test_df[cat_cols].astype('category')

        data_manager = KaggleDatasetManager(train_df,
                                    list(train_df.drop(columns='price').columns),
                                    'price',
                                    0.8,
                                    0.2,
                                    0)
        
        (X_train, 
        X_valid, 
        X_test, 
        y_train, 
        y_valid,
        y_test) = data_manager.dataset_partition()


        lgbm_params = {"num_leaves": 426,
                     "max_depth": 20,
                     "learning_rate": 0.011353178352988012,
                     "num_boost_round": 884,
                     "subsample": 0.5772552201954328,
                     "colsample_bytree": 0.9164865430101521,
                     "reg_alpha": 1.48699088003429e-06,
                     "reg_lambda": 0.41539458543414265,
                     "min_data_in_leaf": 73,
                     "feature_fraction": 0.751673655170548,
                     "bagging_fraction": 0.5120415391590843,
                     "bagging_freq": 2,
                     "min_child_weight": 0.017236362383443497,
                     "cat_smooth": 54.81317407769262,
                     "verbose" : -1,
                     "verbose_eval" : -1,
                     "boosting_type" : "gbdt",
                     "stopping_rounds" : 20}
        
        xgb_params = {"max_bin": 403, 
                     "learning_rate": 0.012720488589018275, 
                     "max_depth": 14, 
                     "num_boost_round": 921, 
                     "gamma": 7.327474792423768, 
                     "min_child_weight": 99.49960880266693, 
                     "subsample": 0.6815290497072164, 
                     "colsample_bytree": 0.6882587387019495, 
                     "colsample_bylevel": 0.6524817277480367, 
                     "colsample_bynode": 0.9692708790975624, 
                     "reg_alpha": 4.444851828081367e-06, 
                     "reg_lambda": 0.9647173450833559, 
                     "max_cat_threshold": 350, 
                     "grow_policy": "lossguide",
                     "stopping_rounds" : 20}
        
        cat_params = {"max_bin": 279, 
                     "learning_rate": 0.05411783467468672, 
                     "depth": 15, 
                     "iterations": 973, 
                     "bagging_temperature": 3.2498902422443727, 
                     "subsample": 0.7472867491636364, 
                     "colsample_bylevel": 0.9612415950869025, 
                     "min_data_in_leaf": 677.4758118043563, 
                     "l2_leaf_reg": 9.475872080718132, 
                     "grow_policy": "Depthwise", 
                     "leaf_estimation_iterations": 1, 
                     "random_strength": 1.153902049647089, 
                     "leaf_estimation_method": "Newton",
                     "stopping_rounds" : 20}
        
        stopping_rounds = 20
        cb = CatBoostModel(**cat_params, verbose=False)
        xgb = XGBoostModel(eval_verbosity=False, **xgb_params)
        lgbm = LGBMModel(**lgbm_params)

        num_splits = 5
        eval_metrics = {"rmse" : lambda y, yhat : root_mean_squared_error(y, yhat)}
        skf = StratifiedKFold(n_splits=num_splits, shuffle=True, random_state=42)

        oofs = []
        test_preds = []
        for model in [cb, xgb, lgbm]:
            cvt = CrossValidateTestSklearnModel(model,
                                        eval_metrics,
                                        skf)

            modellist, cv_scores, test_scores, oof_predictions, test_predictions = cvt.evaluate(X_train,
                                                                                                y_train,
                                                                                                X_test,
                                                                                                y_test,
                                                                                                stratified_set=(y_train > 500000).astype(int))
            test_predictions = np.zeros_like(test_predictions)
            for i in range(len(modellist)):
                test_predictions += modellist[i].predict(X_test)/len(modellist)

            oofs += [oof_predictions[:, None]]
            test_preds += [test_predictions[:, None]]

        meta_model_train = pd.DataFrame(data=np.concatenate(oofs, axis=1), columns=[f"data_{i}" for i in range(len(oofs))])
        meta_model_test = pd.DataFrame(data=np.concatenate(test_preds, axis=1), columns=[f"data_{i}" for i in range(len(oofs))])

        lr = Lasso(alpha=100, positive=True)
        kf = KFold(n_splits=num_splits, shuffle=True, random_state=42)

        cvt = CrossValidateTestSklearnModel(lr,
                                    eval_metrics,
                                    kf)

        modellist, cv_scores, test_scores, oof_predictions, test_predictions = cvt.evaluate(meta_model_train,
                                                                                            y_train,
                                                                                            meta_model_test,
                                                                                            y_test)
        
        final_test_predictions = np.zeros_like(test_predictions)
        for i in range(len(modellist)):
            final_test_predictions += modellist[i].predict(meta_model_test)/len(modellist)

        print("FINAL LR TEST: ", eval_metrics['rmse'](y_test, final_test_predictions))
        
        


            


if __name__ == "__main__":
    MetaModelExperiment().run_used_car_experiment()