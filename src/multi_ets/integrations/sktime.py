"""sktime adapters for ExponentialSmoothing and AutoExponentialSmoothing.

Needs the optional `sktime` extra: `pip install multi-ets[sktime]`.
"""

import copy
from collections.abc import Callable
from typing import Any

import numpy as np

try:
    import pandas as pd
    from sktime.forecasting.base import BaseForecaster
except ImportError as e:
    raise ImportError(
        "multi_ets.integrations.sktime needs sktime: pip install multi-ets[sktime]"
    ) from e

from multi_ets.holt import ExponentialSmoothing
from multi_ets.selection import AutoExponentialSmoothing


class _MultiETSBase(BaseForecaster):
    """Shared fh handling for the two adapters; subclasses build and unwrap the core model.

    fh is given once in fit: its positive relative steps are the training
    horizon steps (a list h for the core), or [insample_step] if it has none.
    In-sample fh points come from the core predict_in_sample(insample_step).
    """

    insample_step: int

    _tags = {  # noqa: RUF012 -- sktime's own convention; ty rejects a ClassVar override
        "y_inner_mtype": "pd.Series",
        "requires-fh-in-fit": True,
        "capability:exogenous": False,
        "capability:missing_values": True,
        "capability:insample": True,
        "capability:pred_int": True,
        "capability:pred_int:insample": True,
    }

    def _make_model(self, steps: list[int]) -> Any:
        raise NotImplementedError

    def _core(self) -> ExponentialSmoothing:
        return self.model_

    def _fit(self, y: Any, X: Any = None, fh: Any = None) -> Any:
        rel = fh.to_relative(self.cutoff).to_numpy()
        steps = [int(s) for s in rel if s > 0] or [self.insample_step]
        self._series = y
        self.model_ = self._make_model(steps).fit(y.to_numpy(dtype=float))
        return self

    def _values(self, fh: Any, alpha: list[float] | None) -> np.ndarray:
        """(len(fh), 1 + n_alphas) point forecast then quantiles, one row per fh point."""
        rel = fh.to_relative(self.cutoff).to_numpy()
        out = np.full((len(rel), 1 + len(alpha or [])), np.nan)
        insample = rel <= 0
        if insample.any():
            core = self._core()
            pos = self._series.index.get_indexer(fh.to_absolute_index(self.cutoff)[insample])
            if alpha is None:
                cols = core.predict_in_sample(self.insample_step)[:, None]
            else:
                cols = np.column_stack(core.predict_in_sample(self.insample_step, alpha))
            out[insample] = cols[pos]
        if not insample.all():
            # Shallow copy with h = the requested steps: predict() must not change fitted
            # state, and the steps shift after update() moves the cutoff.
            core = copy.copy(self._core())
            core.h = [int(s) for s in rel[~insample]]
            out[~insample, 0] = core.predict()
            if alpha is not None:
                out[~insample, 1:] = core.predict_quantiles(alpha)
        return out

    def _predict(self, fh: Any, X: Any = None) -> Any:
        values = self._values(fh, None)[:, 0]
        return pd.Series(values, index=fh.to_absolute_index(self.cutoff), name=self._series.name)

    # sktime calls this with fh=, X=, alpha= keywords; its mixin stub only declares alpha.
    def _predict_quantiles(self, fh: Any, X: Any, alpha: list[float]) -> Any:  # ty: ignore[invalid-method-override]
        values = self._values(fh, list(alpha))[:, 1:]
        columns = self._get_columns(method="predict_quantiles", alpha=alpha)
        return pd.DataFrame(values, index=fh.to_absolute_index(self.cutoff), columns=columns)

    def _update(self, y: Any, X: Any = None, update_params: bool = True) -> Any:
        series = pd.concat([self._series, y])
        self._series = series[~series.index.duplicated(keep="last")]
        if update_params:
            # ponytail: full refit from scratch, warm-start from the old params_ if refits get slow
            return self._fit(self._series, X, self._fh)
        # ponytail: re-filters by swapping the core's private _y; a public refilter method
        # on ExponentialSmoothing if more callers need it
        self._core()._y = self._series.to_numpy(dtype=float)
        return self


class MultiETSForecaster(_MultiETSBase):
    """sktime forecaster for multi_ets.ExponentialSmoothing (joint multi-step loss fit).

    Hyperparameters mirror ExponentialSmoothing, except h: the training horizon
    steps are the positive relative steps of the fh passed to fit (a gapped fh
    like [1, 2, 3, 7, 8, 9] trains on just those steps). A large max(fh) leaves
    few training origins, and the core warns when a step has too few
    residuals for its empirical quantiles.

    Parameters
    ----------
    trend : str or None, default=None
        None, "add" or "damped".
    seasonal_period : int or None, default=None
        Seasonal period, what sktime calls sp.
    seasonal_type : str, default="additive"
        "additive" or "multiplicative".
    loss : str or callable, default="mse"
        Built-in loss name or a callable (y_true, y_pred) -> scalar. A callable
        must be deepcopy-able, since sktime's clone deep-copies hyperparameters.
    n_starts, delta, tau, dispersion, p
        As in ExponentialSmoothing.
    insample_step : int, default=1
        In-sample fh points (relative step <= 0) are the insample_step-ahead
        forecasts of those points, from predict_in_sample(insample_step).

    Notes
    -----
    sample_weight is not available through sktime. y is univariate; sktime
    fits one model per column/series for multivariate or hierarchical y.
    Intervals are empirical in-sample residual quantiles (optimistic); wrap
    in sktime's ConformalIntervals for calibrated conformal intervals. Its
    refits forecast (so train on) every point after the window: use an
    initial_window above half the series and a Period/Datetime-indexed y.

    Examples
    --------
    >>> from sktime.datasets import load_airline
    >>> from multi_ets.integrations.sktime import MultiETSForecaster
    >>> y = load_airline()
    >>> forecaster = MultiETSForecaster(trend="add", seasonal_period=12)
    >>> forecaster.fit(y, fh=[1, 2, 3])
    MultiETSForecaster(...)
    >>> y_pred = forecaster.predict()
    >>> intervals = forecaster.predict_interval(coverage=0.9)
    """

    def __init__(
        self,
        trend: str | None = None,
        seasonal_period: int | None = None,
        seasonal_type: str = "additive",
        loss: str | Callable[..., Any] = "mse",
        n_starts: int = 10,
        delta: float = 1.0,
        tau: float = 0.5,
        dispersion: float = 1.0,
        p: float = 1.5,
        insample_step: int = 1,
    ) -> None:
        self.trend = trend
        self.seasonal_period = seasonal_period
        self.seasonal_type = seasonal_type
        self.loss = loss
        self.n_starts = n_starts
        self.delta = delta
        self.tau = tau
        self.dispersion = dispersion
        self.p = p
        self.insample_step = insample_step
        super().__init__()

    def _make_model(self, steps: list[int]) -> ExponentialSmoothing:
        return ExponentialSmoothing(
            h=steps,
            trend=self.trend,
            seasonal_period=self.seasonal_period,
            seasonal_type=self.seasonal_type,
            loss=self.loss,
            n_starts=self.n_starts,
            delta=self.delta,
            tau=self.tau,
            dispersion=self.dispersion,
            p=self.p,
        )

    @classmethod
    def get_test_params(cls, parameter_set: str = "default") -> list[dict[str, Any]]:
        """Cheap parameter sets for sktime's estimator checks."""
        return [
            {"n_starts": 2},
            {"trend": "damped", "loss": "mae", "n_starts": 1, "insample_step": 2},
        ]


class AutoMultiETSForecaster(_MultiETSBase):
    """sktime forecaster for multi_ets.AutoExponentialSmoothing (CV over trend x seasonal_type).

    Same fh semantics and caveats as MultiETSForecaster; predictions come from
    the winning candidate. CV needs at least n_splits * max(fh) points.

    Parameters
    ----------
    seasonal_period : int or None, default=None
        Seasonal period, what sktime calls sp.
    loss : str or callable, default="mse"
        As in MultiETSForecaster (a callable must be deepcopy-able).
    n_starts, n_splits, delta, tau, dispersion, p
        As in AutoExponentialSmoothing.
    insample_step : int, default=1
        As in MultiETSForecaster.

    Examples
    --------
    >>> from sktime.datasets import load_airline
    >>> from multi_ets.integrations.sktime import AutoMultiETSForecaster
    >>> y = load_airline()
    >>> forecaster = AutoMultiETSForecaster(seasonal_period=12)
    >>> forecaster.fit(y, fh=[1, 2, 3])
    AutoMultiETSForecaster(...)
    >>> y_pred = forecaster.predict()
    """

    def __init__(
        self,
        seasonal_period: int | None = None,
        loss: str | Callable[..., Any] = "mse",
        n_starts: int = 10,
        n_splits: int = 3,
        delta: float = 1.0,
        tau: float = 0.5,
        dispersion: float = 1.0,
        p: float = 1.5,
        insample_step: int = 1,
    ) -> None:
        self.seasonal_period = seasonal_period
        self.loss = loss
        self.n_starts = n_starts
        self.n_splits = n_splits
        self.delta = delta
        self.tau = tau
        self.dispersion = dispersion
        self.p = p
        self.insample_step = insample_step
        super().__init__()

    def _make_model(self, steps: list[int]) -> AutoExponentialSmoothing:
        return AutoExponentialSmoothing(
            h=steps,
            seasonal_period=self.seasonal_period,
            loss=self.loss,
            n_starts=self.n_starts,
            n_splits=self.n_splits,
            delta=self.delta,
            tau=self.tau,
            dispersion=self.dispersion,
            p=self.p,
        )

    def _core(self) -> ExponentialSmoothing:
        return self.model_.best_

    @classmethod
    def get_test_params(cls, parameter_set: str = "default") -> list[dict[str, Any]]:
        """Cheap parameter sets for sktime's estimator checks."""
        return [{"n_starts": 1, "n_splits": 2}, {"loss": "mae", "n_starts": 1, "n_splits": 2}]
