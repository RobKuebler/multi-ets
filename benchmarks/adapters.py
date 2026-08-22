"""Uniform (train, h) -> forecast adapters for multi_ets, statsmodels, statsforecast.

Three families:
  * split_adapters(structure, n_starts) -- fit/predict as separate callables
    (FitPredictAdapter), so callers can time each step independently.
  * matched_adapters(structure) -- primary comparison, same trend/seasonal
    config forced on all three libraries (see CONTEXT.md's "Matched-structure
    comparison"); a thin fit+predict wrapper over split_adapters().
  * autoselect_adapters(seasonal_period) -- bonus comparison, each library
    picks its own structure via its native selection procedure.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from statsforecast.models import AutoETS, SimpleExponentialSmoothingOptimized
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
from statsmodels.tsa.holtwinters import ExponentialSmoothing as SMExponentialSmoothing

from benchmarks.datasets import Structure
from multi_ets import SES, DampedHolt, DampedHoltWinters, Holt, HoltWinters

Adapter = Callable[[np.ndarray, int], np.ndarray]


@dataclass
class FitPredictAdapter:
    """fit(train, h) -> fitted model; predict(fitted, h) -> forecast -- timed separately."""

    fit: Callable[[np.ndarray, int], Any]
    predict: Callable[[Any, int], np.ndarray]


# multi_ets: multi-step joint loss, always. statsmodels: classical SSE,
# 1-step. statsforecast: golden-section SSE for SES, innovations MLE for
# anything with trend/seasonality (see CONTEXT.md's "Innovations ETS" entry
# for why those aren't the same fitting philosophy).
def fitting_method_label(library: str, structure: Structure) -> str:
    if library == "multi_ets":
        return "multi-step SSE"
    if library == "statsmodels":
        return "1-step SSE (classical)"
    if library == "statsforecast":
        if structure.trend == "none" and structure.seasonal == "none":
            return "1-step SSE (golden-section)"
        return "1-step MLE (innovations state-space)"
    raise ValueError(f"Unknown library {library!r}")


def _multi_ets_model(structure: Structure, h: int, n_starts: int = 10):
    if structure.seasonal == "none":
        cls = {"none": SES, "add": Holt, "damped": DampedHolt}[structure.trend]
        return cls(h=h, loss="mse", n_starts=n_starts)
    cls = DampedHoltWinters if structure.damped else HoltWinters
    seasonal_type = "additive" if structure.seasonal == "add" else "multiplicative"
    return cls(
        h=h,
        seasonal_period=structure.seasonal_period,
        seasonal_type=seasonal_type,
        loss="mse",
        n_starts=n_starts,
    )


def _ets_model_string(structure: Structure) -> str:
    trend_letter = "N" if structure.trend == "none" else "A"
    season_letter = {"none": "N", "add": "A", "mul": "M"}[structure.seasonal]
    # Additive-error + multiplicative-seasonality (AAM/ANM/AAdM) is a forbidden
    # combination in the ETS taxonomy (numerically unstable); use multiplicative
    # error there instead. Neither letter has a multi_ets analog either way.
    error_letter = "M" if structure.seasonal == "mul" else "A"
    return f"{error_letter}{trend_letter}{season_letter}"


def split_adapters(structure: Structure, n_starts: int = 10) -> dict[str, FitPredictAdapter]:
    """Same matched-structure models as matched_adapters(), with fit/predict timed separately."""

    def multi_ets_fit(train: np.ndarray, h: int) -> Any:
        return _multi_ets_model(structure, h, n_starts=n_starts).fit(train)

    def multi_ets_predict(fitted: Any, h: int) -> np.ndarray:
        return np.asarray(fitted.predict())

    def statsmodels_fit(train: np.ndarray, h: int) -> Any:
        kwargs: dict = {"initialization_method": "estimated"}
        if structure.trend != "none":
            kwargs["trend"] = "add"
            kwargs["damped_trend"] = structure.damped
        if structure.seasonal != "none":
            kwargs["seasonal"] = structure.seasonal
            kwargs["seasonal_periods"] = structure.seasonal_period
        return SMExponentialSmoothing(train, **kwargs).fit(optimized=True)

    def statsmodels_predict(fitted: Any, h: int) -> np.ndarray:
        return np.asarray(fitted.forecast(h))

    def statsforecast_fit(train: np.ndarray, h: int) -> Any:
        if structure.trend == "none" and structure.seasonal == "none":
            model = SimpleExponentialSmoothingOptimized()
        else:
            model = AutoETS(
                season_length=structure.seasonal_period or 1,
                model=_ets_model_string(structure),
                damped=structure.damped,
            )
        return model.fit(np.asarray(train, dtype=float))

    def statsforecast_predict(fitted: Any, h: int) -> np.ndarray:
        return np.asarray(fitted.predict(h=h)["mean"])

    return {
        "multi_ets": FitPredictAdapter(multi_ets_fit, multi_ets_predict),
        "statsmodels": FitPredictAdapter(statsmodels_fit, statsmodels_predict),
        "statsforecast": FitPredictAdapter(statsforecast_fit, statsforecast_predict),
    }


def one_step_ahead_adapter(structure: Structure, n_starts: int = 10) -> FitPredictAdapter:
    """multi_ets fit like classical exponential smoothing instead of its usual joint loss.

    Fitting uses h=1, so the joint loss collapses to plain one-step-ahead SSE over every
    origin (matching statsmodels' fitting philosophy). Forecasting past step 1 then relies
    on the fitted recursion's own deterministic extrapolation: predict() computes its
    multi-step forecast purely from self.h and the fitted params/states, with no other
    h-dependent state cached by fit(), so retargeting self.h to the desired horizon before
    calling predict() forecasts arbitrarily far ahead from a one-step fit.
    """

    def fit(train: np.ndarray, h: int) -> Any:
        return _multi_ets_model(structure, h=1, n_starts=n_starts).fit(train)

    def predict(fitted: Any, h: int) -> np.ndarray:
        fitted.h = h
        return np.asarray(fitted.predict())

    return FitPredictAdapter(fit, predict)


def matched_adapters(structure: Structure) -> dict[str, Adapter]:
    def combined(adapter: FitPredictAdapter) -> Adapter:
        def fit_predict(train: np.ndarray, h: int) -> np.ndarray:
            return adapter.predict(adapter.fit(train, h), h)

        return fit_predict

    return {library: combined(adapter) for library, adapter in split_adapters(structure).items()}


def _autoselect_candidate_structures(seasonal_period: int | None) -> list[Structure]:
    structures = [Structure("none", "none"), Structure("add", "none"), Structure("damped", "none")]
    if seasonal_period is not None:
        m = seasonal_period
        structures += [
            Structure("add", "add", m),
            Structure("add", "mul", m),
            Structure("damped", "add", m),
            Structure("damped", "mul", m),
        ]
    return structures


def _pick_best_structure(
    fit_and_predict: Callable[[Structure, np.ndarray, int], np.ndarray],
    candidate_structures: list[Structure],
    train: np.ndarray,
    h: int,
) -> Structure:
    """Single in-train holdout (not a full nested CV -- this is a cheap bonus comparison)."""
    if len(train) < 2 * h + 1:
        return candidate_structures[0]  # too short to holdout-validate: fall back to the simplest
    subtrain, subtest = train[:-h], train[-h:]
    best_structure, best_mse = candidate_structures[0], np.inf
    for s in candidate_structures:
        try:
            pred = fit_and_predict(s, subtrain, h)
        except Exception:
            continue
        mse = float(np.mean((np.asarray(pred) - subtest) ** 2))
        if mse < best_mse:
            best_structure, best_mse = s, mse
    return best_structure


def autoselect_adapters(seasonal_period: int | None) -> dict[str, Adapter]:
    candidate_structures = _autoselect_candidate_structures(seasonal_period)

    def multi_ets_fit_predict(train: np.ndarray, h: int) -> np.ndarray:
        def fit_and_predict(s: Structure, tr: np.ndarray, hh: int) -> np.ndarray:
            return _multi_ets_model(s, hh).fit(tr).predict()

        best = _pick_best_structure(fit_and_predict, candidate_structures, train, h)
        return fit_and_predict(best, train, h)

    def _ets_model_kwargs(s: Structure) -> dict:
        kwargs: dict = {"error": "add"}
        if s.trend != "none":
            kwargs["trend"] = "add"
            kwargs["damped_trend"] = s.damped
        if s.seasonal != "none":
            kwargs["seasonal"] = s.seasonal
            kwargs["seasonal_periods"] = s.seasonal_period
        return kwargs

    def statsmodels_fit_predict(train: np.ndarray, h: int) -> np.ndarray:
        def fit_and_predict(s: Structure, tr: np.ndarray, hh: int) -> np.ndarray:
            fit = ETSModel(tr, **_ets_model_kwargs(s)).fit(disp=False)
            return np.asarray(fit.forecast(hh))

        best = _pick_best_structure(fit_and_predict, candidate_structures, train, h)
        return fit_and_predict(best, train, h)

    def statsforecast_fit_predict(train: np.ndarray, h: int) -> np.ndarray:
        model = AutoETS(season_length=seasonal_period or 1, model="ZZZ")
        fitted = model.fit(np.asarray(train, dtype=float))
        return np.asarray(fitted.predict(h=h)["mean"])

    return {
        "multi_ets": multi_ets_fit_predict,
        "statsmodels": statsmodels_fit_predict,
        "statsforecast": statsforecast_fit_predict,
    }
