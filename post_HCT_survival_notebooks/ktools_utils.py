from functools import reduce
from typing import Any, Dict, List, Tuple, Union
import numpy as np
import pandas as pd
from copy import deepcopy
import optuna
from optuna.samplers import TPESampler
from typing import *
from catboost import CatBoostError
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import *
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, MinMaxScaler
import warnings
from enum import Enum
from lifelines import (
    CoxPHFitter,
    ExponentialFitter,
    GeneralizedGammaFitter,
    KaplanMeierFitter,
    NelsonAalenFitter,
    AalenJohansenFitter,
    BreslowFlemingHarringtonFitter,
    WeibullFitter,
    LogLogisticFitter,
    LogNormalFitter,
)
from functools import partial
from scipy.stats import rankdata
from sklearn.preprocessing import quantile_transform


class IModelWrapper(ABC):
    @abstractmethod
    def fit(self):
        pass

    @abstractmethod
    def predict(self):
        pass

    def set_model(self, model):
        self.model = model
        return self


@dataclass
class DataSciencePipelineSettings(object):
    train_csv_path: str
    test_csv_path: str
    target_col_name: List[str]
    original_csv_path: str = None
    original_csv_processing: callable = lambda x: x
    sample_submission_path: str = None
    training_col_names: List[str] = None
    categorical_col_names: List[str] = None
    training_data_percentage: float = 0.8
    category_occurrence_threshold: int = 300
    logged: bool = False

    def __post_init__(self):
        self.train_df, self.test_df = self._load_csv_paths()
        self.training_col_names, self.categorical_col_names = self._get_column_info()
        self.combined_df = self._combine_datasets()

    def _load_csv_paths(self):
        train_df = self._smart_drop_index(pd.read_csv(self.train_csv_path))
        test_df = self._smart_drop_index(pd.read_csv(self.test_csv_path))
        if self.original_csv_path is not None:
            train_df = train_df.assign(source=0)
            test_df = test_df.assign(source=0)
            original_df = self._smart_drop_index(
                pd.read_csv(self.original_csv_path)
            ).assign(source=1)
            original_df = self.original_csv_processing(original_df)

            pd.testing.assert_index_equal(
                train_df.columns.sort_values(),
                original_df.columns.sort_values(),
                check_exact=True,
            )
            pd.testing.assert_series_equal(
                train_df.dtypes.sort_index(),
                original_df.dtypes.sort_index(),
                check_exact=True,
            )
            train_df = pd.concat([train_df, original_df], axis=0).reset_index(drop=True)

        return train_df, test_df

    def _get_column_info(self):
        cat_col_names = [
            col_name
            for col_name in self.train_df.columns
            if self.train_df[col_name].dtype == "object"
        ]
        training_features = list(
            self.train_df.drop(columns=self.target_col_name).columns
        )
        cat_col_names = (
            cat_col_names
            if self.categorical_col_names is None
            else self.categorical_col_names
        )
        return training_features, cat_col_names

    def _combine_datasets(self):
        combined_df = pd.concat([self.train_df, self.test_df], keys=["train", "test"])
        return combined_df

    def update(self):
        self.train_df = self.combined_df.loc["train"].copy()
        self.test_df = self.combined_df.loc["test"].copy()
        return self.train_df, self.test_df

    @staticmethod
    def _smart_drop_index(df):
        try:
            differences = df.iloc[:, 0].diff().dropna()
            if differences.nunique() == 1:
                df = df.drop(columns=df.columns[0])
        except:
            pass
        return df

    @property
    def target_col(self):
        """target column name property."""
        return self.target_col_name

    @target_col.setter
    def target_col(self, value):
        self.target_col_name = value


class ConvertToLower:
    @staticmethod
    def transform(original_settings: DataSciencePipelineSettings):
        settings = deepcopy(original_settings)
        for col_name in settings.categorical_col_names:
            settings.combined_df[col_name] = settings.combined_df[col_name].str.lower()
        return settings


class FillNullValues:
    @staticmethod
    def transform(
        original_settings: DataSciencePipelineSettings,
        numeric_fill=-1,
        category_fill="missing",
    ):
        settings = deepcopy(original_settings)
        for col_name in settings.training_col_names:
            if pd.api.types.is_numeric_dtype(settings.combined_df[col_name]):
                settings.combined_df[col_name] = settings.combined_df[col_name].fillna(
                    numeric_fill
                )
            else:
                settings.combined_df[col_name] = settings.combined_df[col_name].fillna(
                    category_fill
                )
        return settings


class ConvertObjectToCategorical:
    @staticmethod
    def transform(original_settings: DataSciencePipelineSettings):
        settings = deepcopy(original_settings)
        cat_cols = settings.categorical_col_names
        settings.combined_df[cat_cols] = settings.combined_df[cat_cols].astype(
            "category"
        )
        return settings


class ConvertObjectToStrCategorical:
    @staticmethod
    def transform(original_settings: DataSciencePipelineSettings):
        settings = deepcopy(original_settings)
        cat_cols = settings.categorical_col_names
        settings.combined_df[cat_cols] = (
            settings.combined_df[cat_cols].astype(str).astype("category")
        )
        return settings


class ConvertAllToCategorical:
    @staticmethod
    def transform(original_settings: DataSciencePipelineSettings):
        settings = deepcopy(original_settings)
        all_cols = settings.training_col_names
        settings.combined_df[all_cols] = (
            settings.combined_df[all_cols].astype(str).astype("category")
        )
        return settings


class LogTransformTarget:
    @staticmethod
    def transform(original_settings: DataSciencePipelineSettings):
        settings = deepcopy(original_settings)
        target = settings.target_col_name
        settings.combined_df[target] = np.log1p(settings.combined_df[target])
        return settings


class OrdinalEncode:
    @staticmethod
    def transform(original_settings: DataSciencePipelineSettings):
        settings = deepcopy(original_settings)
        train_df, test_df = settings.update()
        ordinal_encoder = OrdinalEncoder(
            encoded_missing_value=-1,
            handle_unknown="use_encoded_value",
            unknown_value=-1,
        )
        train_df[settings.categorical_col_names] = ordinal_encoder.fit_transform(
            train_df[settings.categorical_col_names]
        )
        test_df[settings.categorical_col_names] = ordinal_encoder.transform(
            test_df[settings.categorical_col_names]
        )
        settings.combined_df = pd.concat([train_df, test_df], keys=["train", "test"])
        settings.combined_df[settings.categorical_col_names] = settings.combined_df[
            settings.categorical_col_names
        ].astype(int)
        return settings


class StandardScaleNumerical:
    @staticmethod
    def transform(original_settings: DataSciencePipelineSettings):
        settings = deepcopy(original_settings)
        scaler = StandardScaler()
        train_df, test_df = settings.update()
        num_cols = settings.combined_df.select_dtypes(include=["number"]).columns
        train_df[num_cols] = scaler.fit_transform(train_df[num_cols])
        test_df[num_cols] = scaler.transform(test_df[num_cols])
        settings.combined_df = pd.concat([train_df, test_df], keys=["train", "test"])
        return settings


class MinMaxScalerNumerical:
    @staticmethod
    def transform(original_settings: DataSciencePipelineSettings):
        settings = deepcopy(original_settings)
        scaler = MinMaxScaler()
        train_df, test_df = settings.update()
        num_cols = settings.combined_df.select_dtypes(include=["number"]).columns
        train_df[num_cols] = scaler.fit_transform(train_df[num_cols])
        test_df[num_cols] = scaler.transform(test_df[num_cols])
        settings.combined_df = pd.concat([train_df, test_df], keys=["train", "test"])
        return settings


class CrossValidationExecutor:
    def __init__(
        self,
        sklearn_model_instance,
        evaluation_metric,
        kfold_object,
        training_features: Union[List[str], None] = None,
        use_test_as_valid=True,
        num_classes=None,
        verbose=1,
    ) -> None:
        self.model = sklearn_model_instance
        self._evaluation_metric = evaluation_metric
        self._kf = kfold_object
        self._num_splits = kfold_object.get_n_splits()
        self._training_features = training_features
        self._use_test_as_valid = use_test_as_valid
        self._num_classes = num_classes
        self._verbose = verbose

    def run(
        self,
        X: pd.DataFrame,
        y: Union[pd.DataFrame, pd.Series],
        weights=None,
        test_data=None,
        groups=None,
        additional_data=None,
        local_transform_list=[lambda x: x],
        output_transform_list=[lambda x: x[-1]],
    ) -> Tuple[Tuple[float], np.ndarray, List[Any]]:
        training_features = (
            X.columns.tolist()
            if self._training_features is None
            else self._training_features
        )
        X.reset_index(drop=True, inplace=True)
        y.reset_index(drop=True, inplace=True)

        if additional_data is not None:
            X_add, y_add = additional_data
            pd.testing.assert_index_equal(X.columns, X_add.columns, check_exact=True)
            pd.testing.assert_series_equal(X.dtypes, X_add.dtypes, check_exact=True)
            pd.testing.assert_index_equal(y.columns, y_add.columns, check_exact=True)
            pd.testing.assert_series_equal(y.dtypes, y_add.dtypes, check_exact=True)

        cv_results = []
        model_list = []
        oof_predictions = None
        metric_predictions = None
        test_predictions = None

        groups = y if groups is None else groups
        weights = np.ones(y.shape[0]) if weights is None else weights

        for i, (train_index, val_index) in enumerate(
            self._kf.split(X, groups, groups=groups)
        ):
            X_full_test = X.loc[val_index, :]
            X_train, X_test = (
                X.loc[train_index, training_features],
                X.loc[val_index, training_features],
            )
            y_train, y_test = y.loc[train_index], y.loc[val_index]
            train_weights = weights[train_index]

            if additional_data is not None:
                X_train = pd.concat([X_train, X_add], axis=0)
                y_train = pd.concat([y_train, y_add], axis=0)

            X_train, y_train = reduce(
                lambda acc, func: func(acc), local_transform_list, (X_train, y_train)
            )
            validation_set = None
            if self._use_test_as_valid:
                validation_set = [X_test, y_test]

            model = deepcopy(self.model).fit(
                X_train, y_train, validation_set=validation_set, weights=train_weights
            )
            model_list += [model]
            y_pred = model.predict(X_test)
            y_pred_processed = reduce(
                lambda acc, func: func(acc),
                output_transform_list,
                (X_full_test.copy(), y_pred),
            )

            cv_results += [self._evaluation_metric(y_test, deepcopy(y_pred_processed))]

            if oof_predictions is None:
                oof_shape = (
                    (y.shape[0],)
                    if len(y_pred.shape) == 1
                    else (y.shape[0], y_pred.shape[-1])
                )
                oof_predictions = np.zeros(oof_shape)
            if metric_predictions is None:
                y_hat_shape = (
                    (y.shape[0],)
                    if len(y_pred_processed.shape) == 1
                    else (y.shape[0], y_pred_processed.shape[-1])
                )
                metric_predictions = np.zeros(y_hat_shape)
            if test_data is not None:
                test_preds = model.predict(test_data)
                if test_predictions is None:
                    test_predictions = test_preds / self._num_splits
                else:
                    test_predictions += test_preds / self._num_splits

            oof_predictions[val_index] = y_pred
            metric_predictions[val_index] = y_pred_processed

            if self._verbose > 1:
                print(f"The CV results of the current fold is {cv_results[-1]}")

        oof_score = self._evaluation_metric(y, metric_predictions)
        mean_cv_score = np.mean(cv_results)
        score_tuple = (oof_score, mean_cv_score)

        if self._verbose > 0:
            print("#" * 100)
            print("OOF prediction score : ", oof_score)
            print(
                f"Mean {self._num_splits}-cv results : {mean_cv_score} +- {np.std(cv_results)}"
            )
            print("#" * 100)

        return score_tuple, oof_predictions, model_list, test_predictions


class OptunaHyperparameterOptimizer:
    def __init__(
        self,
        X_train: pd.DataFrame,
        y_train: pd.DataFrame,
        model,
        param_grid_getter,
        kfold_object,
        metric: callable,
        fixed_model_params={},
        model_wrapper: IModelWrapper = None,
        training_features: Union[List[str], None] = None,
        direction: str = "maximize",
        n_trials: int = 100,
        study_name: str = "ml_experiment",
        explore_fraction: float = 0.1,
        cross_validation_run_kwargs={},
        verbose=False,
        random_state=42,
    ) -> None:
        super().__init__()
        self._X_train = X_train
        self._y_train = y_train
        self.model = model
        self._metric = metric
        self._fixed_model_params = fixed_model_params
        self._model_wrapper = model_wrapper
        self._training_features = training_features
        self._param_grid_getter = param_grid_getter
        self._kfold_object = kfold_object
        self._direction = direction
        self._n_trials = n_trials
        self._study_name = study_name
        self._explore_fraction = explore_fraction
        self._cross_validation_run_kwargs = cross_validation_run_kwargs
        self._verbose = verbose
        self._random_state = random_state

    def optimize(
        self,
        inital_parameters: Dict[str, float] = None,
        initial_distribution: Dict[str, Any] = None,
        timeout: int = 3600,
    ):
        if self._verbose:
            print("#" * 100)
            print("Starting Optuna Optimizer")
            print("#" * 100)

        sampler = TPESampler(
            n_startup_trials=int(self._n_trials * self._explore_fraction),
            seed=self._random_state,
        )

        storage_name = "sqlite:///{}.db".format(self._study_name)
        study = optuna.create_study(
            sampler=sampler,
            study_name=self._study_name,
            direction=self._direction,
            storage=storage_name,
            load_if_exists=True,
        )

        if inital_parameters is not None:
            fixed_trial = optuna.trial.FixedTrial(inital_parameters)
            study.add_trial(
                optuna.create_trial(
                    params=inital_parameters,
                    distributions=initial_distribution,
                    value=self._objective(fixed_trial),
                )
            )
        study.optimize(
            self._objective,
            n_trials=self._n_trials,
            timeout=timeout,
            catch=(CatBoostError,),
        )
        optimal_params = study.best_params
        optimal_params.update(**self._fixed_model_params)
        return optimal_params

    def _objective(self, trial: optuna.Trial):
        parameters = self._param_grid_getter.get(trial)

        parameters = self._model_wrapper.take_params(parameters)
        model = self.model(**parameters, **self._fixed_model_params)
        if self._model_wrapper is not None:
            model = self._model_wrapper.set_model(model)

        cv_scores, oof, model_list, _ = CrossValidationExecutor(
            model,
            self._metric,
            self._kfold_object,
            training_features=self._training_features,
            use_test_as_valid=True,
        ).run(self._X_train, self._y_train, **self._cross_validation_run_kwargs)

        return cv_scores[0]


def transform_kaplan_meier(time, event, separation: float = 0.0):
    kmf = KaplanMeierFitter()
    kmf.fit(time, event)
    y = kmf.survival_function_at_times(time).values
    y = np.where(event == 0, y - separation, y)
    return y


def transform_nelson_aalen(time, event, separation: float = 0.0):
    naf = NelsonAalenFitter()
    naf.fit(time, event)
    y = -naf.cumulative_hazard_at_times(time).values
    y = np.where(event == 0, y - separation, y)
    return y


def transform_partial_hazard(time, event, separation: float = 0.0):
    data = pd.DataFrame({"efs_time": time, "efs": event, "time": time, "event": event})
    cph = CoxPHFitter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cph.fit(data, duration_col="time", event_col="event")
    y = cph.predict_partial_hazard(data)
    y = np.where(event == 0, y - separation, y)
    return y


def transform_separate(time, event):
    transformed = time.values.copy()
    mx = transformed[event == 1].max()
    mn = transformed[event == 0].min()
    transformed[event == 0] = time[event == 0] + mx - mn
    transformed = rankdata(transformed)
    transformed[event == 0] += len(transformed) // 2
    transformed = transformed / transformed.max()
    return -transformed


def transform_rank_log(time, event):
    transformed = time.values.copy()
    mx = transformed[event == 1].max()
    mn = transformed[event == 0].min()
    transformed[event == 0] = time[event == 0] + mx - mn
    transformed = rankdata(transformed)
    transformed[event == 0] += len(transformed) * 2
    transformed = transformed / transformed.max()
    transformed = np.log(transformed)
    return -transformed


def transform_quantile(time, event, separation=0.3):
    transformed = np.full(len(time), np.nan)
    transformed_dead = quantile_transform(
        -time[event == 1].values.reshape(-1, 1)
    ).ravel()
    transformed[event == 1] = transformed_dead
    transformed[event == 0] = transformed_dead.min() - separation
    return transformed


def transform_aalen_johansen(time, event, separation: float = 0.0):
    ajf = AalenJohansenFitter()
    ajf.fit(time.values, event, event_of_interest=1)
    y = 1 - ajf.predict(time.values).values
    y = np.where(event == 0, y - separation, y)
    return y


def transform_breslow_flemingharrington(time, event, separation: float = 0.0):
    bfh = BreslowFlemingHarringtonFitter()
    bfh.fit(time.values, event)
    y = bfh.predict(time.values).values
    y = np.where(event == 0, y - separation, y)
    return y


def transform_exponential(time, event, separation: float = 0.0):
    ef = ExponentialFitter()
    ef.fit(time, event)
    y = ef.survival_function_at_times(time.values).values
    y = np.where(event == 0, y - separation, y)
    return y


def transform_generalized_gamma(time, event, separation: float = 0.0):
    ggf = GeneralizedGammaFitter()
    ggf.fit(time, event)
    y = ggf.survival_function_at_times(time.values).values
    y = np.where(event == 0, y - separation, y)
    return y


def transform_weibull(time, event, separation: float = 0.0):
    wf = WeibullFitter()
    wf.fit(time, event)
    y = wf.survival_function_at_times(time.values).values
    y = np.where(event == 0, y - separation, y)
    return y


def transform_log_normal(time, event, separation: float = 0.0):
    lnf = LogNormalFitter()
    lnf.fit(time, event)
    y = lnf.survival_function_at_times(time.values).values
    y = np.where(event == 0, y - separation, y)
    return y


def transform_log_logistic(time, event, separation: float = 0.0):
    llf = LogLogisticFitter()
    llf.fit(time, event)
    y = llf.survival_function_at_times(time.values).values
    y = np.where(event == 0, y - separation, y)
    return y


def transform_cox(time: np.ndarray, event: np.ndarray):
    return np.where(event.astype(bool), time, -time)


def transform_aft(time: np.ndarray, event: np.ndarray):
    time = time.squeeze()
    lower_bound = time[:, None]
    upper_bound = np.where(event == 0, np.inf, time)[:, None]
    bounds = np.concatenate([lower_bound, upper_bound], axis=1)
    return bounds


class SupportedSurvivalTransformation(Enum):
    QUANTILE = partial(transform_quantile)
    COXPH = partial(transform_partial_hazard)
    SEPARATE = partial(transform_separate)
    RANKLOG = partial(transform_rank_log)
    KAPLANMEIER = partial(transform_kaplan_meier)
    NELSONAALEN = partial(transform_nelson_aalen)
    AALENJOHANSEN = partial(transform_aalen_johansen)
    BRESLOWFLEMINGHARRINGTON = partial(transform_breslow_flemingharrington)
    EXPONENTIAL = partial(transform_exponential)
    GENERALIZEDGAMMA = partial(transform_generalized_gamma)
    WEIBULL = partial(transform_weibull)
    LOGNORMAL = partial(transform_log_normal)
    LOGLOGISTIC = partial(transform_log_logistic)
    COX = partial(transform_cox)
    AFT = partial(transform_aft)


class SurvivalModelWrapper(IModelWrapper):
    def __init__(
        self,
        transform_string: str,
        times_col: str = "efs_time",
        event_col: str = "efs",
        **transform_kwargs,
    ) -> None:
        super().__init__()
        self.transform = SupportedSurvivalTransformation[transform_string.upper()].value
        self._times_col = times_col
        self._event_col = event_col
        self._transform_kwargs = transform_kwargs

    def set_model(self, model):
        self.model = model
        return self

    def take_params(self, params: Dict[str, Any]):
        for k, v in params.items():
            if "surv" in k:
                v = params.pop(k)
                self._transform_kwargs.update({k: v})
        return params

    def fit(
        self, X: pd.DataFrame, y: pd.DataFrame, *args, validation_set=None, **kwargs
    ):
        X_valid, y_valid = validation_set

        valid_size = y_valid.shape[0]
        full_y = pd.concat([y_valid, y])

        y_full_trans = self.transform(
            full_y[self._times_col], full_y[self._event_col], **self._transform_kwargs
        )
        y_valid = y_full_trans[:valid_size]

        y_trans = self.transform(
            y[self._times_col], y[self._event_col], **self._transform_kwargs
        )
        self.model.fit(X, y_trans, *args, validation_set=[X_valid, y_valid], **kwargs)
        return self

    def predict(self, X, *args, **kwargs):
        y_pred = self.model.predict(X, *args, **kwargs)
        return y_pred
