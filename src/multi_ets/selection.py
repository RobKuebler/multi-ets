"""Model selection via rolling-origin cross-validation."""

from collections.abc import Callable, Iterable
from typing import Any, Self

import numpy as np

from multi_ets.holt import ExponentialSmoothing, _loss_key, _loss_value_np, _nan_actual_mask


def select_model(
    candidates: Iterable[ExponentialSmoothing], y: Any, n_splits: int
) -> tuple[ExponentialSmoothing, dict[ExponentialSmoothing, float]]:
    """Pick the best of `candidates` by rolling-origin CV, each scored with its own loss/h.

    Split scheme: rolling-origin, expanding-window, non-overlapping, step =
    candidate's own h. Split i (earliest first) trains on
    y[:len(y) - (n_splits - i) * h] and scores against the next h points.

    Returns (best, scores): best is refit on the full series, scores is
    {candidate: mean_cv_loss} for every candidate.
    """
    y = np.asarray(y, dtype=float)
    scores: dict[ExponentialSmoothing, float] = {}
    for candidate in candidates:
        loss_key = _loss_key(
            candidate.loss, candidate.delta, candidate.tau, candidate.alpha, candidate.p
        )
        h = candidate.h
        if len(y) < n_splits * h:
            raise ValueError(
                f"Series too short for n_splits={n_splits} with h={h}: "
                f"need at least {n_splits * h} points, got {len(y)}"
            )
        losses = []
        for split in range(n_splits):
            cutoff = len(y) - (n_splits - split) * h
            train, test = y[:cutoff], y[cutoff : cutoff + h]
            candidate.fit(train)
            pred = candidate.predict()
            loss = (
                loss_key(test, pred)
                if callable(loss_key)
                else _loss_value_np(loss_key, test, pred, actual_mask=_nan_actual_mask(test))
            )
            losses.append(float(loss))
        scores[candidate] = float(np.mean(losses))

    best = min(scores, key=lambda c: scores[c])
    best.fit(y)
    return best, scores


class AutoExponentialSmoothing:
    """Auto-selects a trend x seasonal_type ExponentialSmoothing config via CV.

    .fit(y) builds every valid (trend, seasonal_type) candidate at the given
    seasonal_period -- trend in (None, "add", "damped"), each combined with
    "additive"/"multiplicative" seasonal_type when seasonal_period is set, or
    just the three trend options with no seasonal candidates when it's None
    -- and picks the best via select_model. A thin wrapper: all CV/scoring
    logic lives in select_model, not duplicated here.

    Named AutoExponentialSmoothing rather than AutoETS: AutoETS is
    statsforecast's name for an innovations-based (MLE) auto-selector, and
    this repo's own benchmarks compare against it directly -- reusing that
    name for a classical auto-selector would collide with CONTEXT.md's rule
    to always qualify bare "ETS" as classical/innovations.
    """

    def __init__(
        self,
        h: int,
        seasonal_period: int | None = None,
        loss: str | Callable[..., Any] = "mse",
        n_starts: int = 10,
        n_splits: int = 3,
        delta: float = 1.0,
        tau: float = 0.5,
        alpha: float = 1.0,
        p: float = 1.5,
    ) -> None:
        self.h = h
        self.seasonal_period = seasonal_period
        self.loss = loss
        self.n_starts = n_starts
        self.n_splits = n_splits
        self.delta = delta
        self.tau = tau
        self.alpha = alpha
        self.p = p
        self.best_: ExponentialSmoothing | None = None
        self.scores_: dict[ExponentialSmoothing, float] | None = None

    def _candidate(self, trend: str | None, seasonal_type: str) -> ExponentialSmoothing:
        return ExponentialSmoothing(
            h=self.h,
            trend=trend,
            seasonal_period=self.seasonal_period,
            seasonal_type=seasonal_type,
            loss=self.loss,
            n_starts=self.n_starts,
            delta=self.delta,
            tau=self.tau,
            alpha=self.alpha,
            p=self.p,
        )

    def _candidates(self) -> list[ExponentialSmoothing]:
        trends: list[str | None] = [None, "add", "damped"]
        if self.seasonal_period is None:
            return [self._candidate(trend, "additive") for trend in trends]
        return [
            self._candidate(trend, seasonal_type)
            for trend in trends
            for seasonal_type in ("additive", "multiplicative")
        ]

    def fit(self, y: Any) -> Self:
        """Pick the best candidate by rolling-origin CV (see select_model); returns self."""
        self.best_, self.scores_ = select_model(self._candidates(), y, self.n_splits)
        return self

    def predict(self) -> np.ndarray:
        """Delegate to the winning candidate's predict()."""
        if self.best_ is None:
            raise ValueError("Model is not fitted yet; call .fit(y) first")
        return self.best_.predict()

    def predict_interval(
        self, coverage: float | list[float] | tuple[float, ...] = 0.9
    ) -> tuple[np.ndarray, np.ndarray] | dict[float, tuple[np.ndarray, np.ndarray]]:
        """Delegate to the winning candidate's predict_interval()."""
        if self.best_ is None:
            raise ValueError("Model is not fitted yet; call .fit(y) first")
        return self.best_.predict_interval(coverage)
