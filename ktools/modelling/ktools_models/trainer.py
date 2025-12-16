from enum import Enum
from functools import reduce
from typing import Any, Dict, List
from ktools.fitting.cross_validation_executor import CrossValidationExecutor
from ktools.hyperparameter_optimization.i_sklearn_kfold_object import (
    ISklearnKFoldObject,
)
from ktools.models import CatBoostModel
from ktools.modelling.ktools_models.hgb_model import HGBModel
from ktools.models import LGBMModel
from ktools.models import XGBoostModel
from ktools.modelling.ktools_models.yggdrasil_gbt_model import YDFGBoostModel
from ktools.preprocessing.basic_feature_transformers import *
from ktools.utils.data_science_pipeline_settings import DataSciencePipelineSettings


class SupportedModelTypes(Enum):
    LGBM = LGBMModel
    CAT = CatBoostModel
    XGB = XGBoostModel
    HGB = HGBModel
    YDF = YDFGBoostModel
    # PYTORCH = PytorchFFNModel
    # KERAS_EMB = KerasEmbeddingModel
    # KERAS_FM = KerasFM
    # TABNET = TabNetModel


class SupportedClassificationParams(Enum):
    LGBM = {"objective": "binary", "metric": "binary_logloss"}
    CAT = {"loss_function": "Logloss", "eval_metric": "AUC"}
    XGB = {"objective": "binary:logistic", "eval_metric": "logloss"}
    HGB = {"target_type": "binary"}
    YDF = {"task": "CLASSIFICATION", "loss": "BINOMIAL_LOG_LIKELIHOOD"}
    # PYTORCH = {}
    # KERAS_EMB = {}
    # KERAS_FM = {}
    # TABNET =  {}


class SupportedRegressionParams(Enum):
    LGBM = {"objective": "regression", "metric": "rmse"}
    CAT = {"loss_function": "RMSE", "eval_metric": "RMSE"}
    XGB = {"objective": "reg:squarederror", "eval_metric": "rmse"}
    HGB = {"target_type": "continuous"}
    YDF = {"task": "REGRESSION", "loss": "SQUARED_ERROR"}
    # PYTORCH = {"loss": nn.MSELoss(), "metric_callable": mean_squared_error}
    # KERAS_EMB = {}
    # KERAS_FM = {}
    # TABNET =  {}


class KToolsTrainer:
    def __init__(
        self,
        model_type: str,
        task: str,
        model_parameters: Dict[str, Any],
        kfold_object: ISklearnKFoldObject,
        train_csv_path: str,
        test_csv_path: str,
        sample_csv_path: str,
        target_col_name: str,
        model_name: str = None,
        output_file_path: str = None,
        data_transforms: List[Any] = [
            FillNullValues.transform,
            ConvertObjectToCategorical.transform,
        ],
        eval_metric: callable = None,
        verbose: bool = False,
    ) -> None:
        self._model_type = model_type.upper()
        self.model_name = model_name
        self._task = task.upper()
        self._model_parameters = model_parameters
        self._kfold_object = kfold_object
        self._eval_metric = eval_metric
        self._verbose = verbose
        self._data_transforms = data_transforms
        self._train_csv_path = train_csv_path
        self._test_csv_path = test_csv_path
        self._sample_csv_path = sample_csv_path
        self._target_col_name = target_col_name
        self._output_file_path = output_file_path
        self.model = self._setup_model()
        self.train_df, self.test_df = self._setup_dataset()

    def _setup_model(self):
        model_class_obj = SupportedModelTypes[self._model_type].value
        if self._task == "BINARY":
            task_params = SupportedClassificationParams[self._model_type].value
        elif self._task == "REGRESSION":
            task_params = SupportedRegressionParams[self._model_type].value
        else:
            raise NotImplementedError

        self._model_parameters.update(task_params)
        return model_class_obj(**self._model_parameters)

    def _setup_dataset(self):
        settings = DataSciencePipelineSettings(
            self._train_csv_path,
            self._test_csv_path,
            self._target_col_name,
        )

        settings = reduce(lambda acc, func: func(acc), self._data_transforms, settings)
        train_df, test_df = settings.update()
        test_df.drop(columns=[self._target_col_name], inplace=True)
        return train_df, test_df

    def fit_predict(self):
        X, y = (
            self.train_df.drop(columns=self._target_col_name),
            self.train_df[[self._target_col_name]],
        )
        score_tuple, oof_predictions, model_list = CrossValidationExecutor(
            self.model, self._eval_metric, self._kfold_object, verbose=2
        ).run(X, y)

        num_splits = self._kfold_object.get_n_splits()
        test_predictions = np.zeros(self.test_df.shape[0])
        for model in model_list:
            test_predictions += model.predict(self.test_df) / num_splits

        self.model_name = (
            str(self.model) if self.model_name is None else self.model_name
        )
        if self._output_file_path is not None:
            pd.Series(oof_predictions).to_csv(
                self._output_file_path + self.model_name + "_oofs.csv"
            )
            pd.Series(test_predictions).to_csv(
                self._output_file_path + self.model_name + "_test.csv"
            )

            sample_sub = pd.read_csv(self._sample_csv_path)
            sample_sub.iloc[:, 1] = test_predictions
            sample_sub.to_csv(f"{self.model_name}_submission.csv", index=False)
            sample_sub.head()
