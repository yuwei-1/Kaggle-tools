import numpy as np
import pandas as pd
from ktools.hyperparameter_optimization.i_sklearn_kfold_object import ISklearnKFoldObject
from ktools.modelling.Interfaces.i_sklearn_model import ISklearnModel


def create_oofs_from_model(cross_validation_executor,
                           X_train,
                           y_train,
                           X_test,
                           additional_data = None,
                           model_string : str = None,
                           directory_path : str = None,
                           sample_submission_file : str = None
                           ):
    score_tuple, oof_predictions, model_list = cross_validation_executor.run(X_train, y_train, additional_data=additional_data)
    num_splits = cross_validation_executor._num_splits

    test_predictions = np.zeros(X_test.shape[0])
    for model in model_list:
        test_predictions += model.predict(X_test)/num_splits

    model_string = str(cross_validation_executor.model) if model_string is None else model_string
    if directory_path is not None:
        pd.Series(oof_predictions).to_csv(directory_path + model_string + "_oofs.csv")
        pd.Series(test_predictions).to_csv(directory_path + model_string + "_test.csv")

        if sample_submission_file is not None:
            sample_sub = pd.read_csv(sample_submission_file)
            sample_sub.iloc[:, 1] =  test_predictions
            sample_sub.to_csv(f"{model_string}_submission.csv", index=False)
            sample_sub.head()
    
    return score_tuple, oof_predictions, test_predictions, model_list