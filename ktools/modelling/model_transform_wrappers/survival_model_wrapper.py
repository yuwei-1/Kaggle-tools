import warnings
import numpy as np
from enum import Enum
from lifelines import *
import pandas as pd
from functools import partial
from scipy.stats import rankdata
from typing import *
from sklearn.preprocessing import quantile_transform
from ktools.modelling.Interfaces.i_model_wrapper import IModelWrapper
from ktools.modelling.Interfaces.i_ktools_model import IKtoolsModel


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

    def set_model(self, model: IKtoolsModel):
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
