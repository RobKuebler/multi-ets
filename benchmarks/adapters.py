"""Uniform (train, h) -> forecast adapters for multi_ets, statsmodels, statsforecast.

Two families:
  * matched_adapters(structure)   -- primary comparison, same trend/seasonal
    config forced on all three libraries (see CONTEXT.md's "Matched-structure
    comparison").
  * autoselect_adapters(seasonal_period) -- bonus comparison, each library
    picks its own structure via its native selection procedure.
"""

from collections.abc import Callable

import numpy as np
from statsforecast.models import AutoETS, SimpleExponentialSmoothingOptimized
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
from statsmodels.tsa.holtwinters import ExponentialSmoothing as SMExponentialSmoothing

from benchmarks.datasets import Structure
from multi_ets import SES, DampedHolt, DampedHoltWinters, Holt, HoltWinters

Adapter = Callable[[np.ndarray, int], np.ndarray]


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


def _multi_ets_model(structure: Structure, h: int):
    if structure.seasonal == "none":
        cls = {"none": SES, "add": Holt, "damped": DampedHolt}[structure.trend]
        return cls(h=h, loss="mse")
    cls = DampedHoltWinters if structure.damped else HoltWinters
    seasonal_type = "additive" if structure.seasonal == "add" else "multiplicative"
    return cls(
        h=h, seasonal_period=structure.seasonal_period, seasonal_type=seasonal_type, loss="mse"
    )


def _ets_model_string(structure: Structure) -> str:
    trend_letter = "N" if structure.trend == "none" else "A"
    season_letter = {"none": "N", "add": "A", "mul": "M"}[structure.seasonal]
    # Additive-error + multiplicative-seasonality (AAM/ANM/AAdM) is a forbidden
    # combination in the ETS taxonomy (numerically unstable); use multiplicative
    # error there instead. Neither letter has a multi_ets analog either way.
    error_letter = "M" if structure.seasonal == "mul" else "A"
    return f"{error_letter}{trend_letter}{season_letter}"


def matched_adapters(structure: Structure) -> dict[str, Adapter]:
    def multi_ets_fit_predict(train: np.ndarray, h: int) -> np.ndarray:
        return _multi_ets_model(structure, h).fit(train).predict()

    def statsmodels_fit_predict(train: np.ndarray, h: int) -> np.ndarray:
        kwargs: dict = {"initialization_method": "estimated"}
        if structure.trend != "none":
            kwargs["trend"] = "add"
            kwargs["damped_trend"] = structure.damped
        if structure.seasonal != "none":
            kwargs["seasonal"] = structure.seasonal
            kwargs["seasonal_periods"] = structure.seasonal_period
        fit = SMExponentialSmoothing(train, **kwargs).fit(optimized=True)
        return np.asarray(fit.forecast(h))

    def statsforecast_fit_predict(train: np.ndarray, h: int) -> np.ndarray:
        if structure.trend == "none" and structure.seasonal == "none":
            model = SimpleExponentialSmoothingOptimized()
        else:
            model = AutoETS(
                season_length=structure.seasonal_period or 1,
                model=_ets_model_string(structure),
                damped=structure.damped,
            )
        fitted = model.fit(np.asarray(train, dtype=float))
        return np.asarray(fitted.predict(h=h)["mean"])

    return {
        "multi_ets": multi_ets_fit_predict,
        "statsmodels": statsmodels_fit_predict,
        "statsforecast": statsforecast_fit_predict,
    }


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
