"""Holt's linear trend model, fit by minimizing a joint h-step-ahead loss."""

import functools
import itertools
from collections.abc import Callable
from typing import Any, Self

import jax
import jax.numpy as jnp
import jaxopt
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm

jax.config.update("jax_enable_x64", True)

# Arrays flow through this module as a duck-typed mix of Python scalars, numpy
# arrays, and jax arrays (jnp.asarray()'d at the boundaries); `Any` reflects
# that rather than pretending to a precision the code doesn't have.
Array = jax.Array

_BOUND_EPS = 1e-4
# Standard classical-exponential-smoothing damping-parameter range (Hyndman &
# Athanasopoulos); phi
# approaching 1 means "no damping", which defeats the point of a damped model.
_PHI_UPPER = 0.98
# L-BFGS-B stopping criteria for the vmapped multi-start solver below. A low
# maxiter is deliberate, not just a speed knob: jaxopt's LBFGSB converges
# more thoroughly per-start than scipy's did, so a generous budget lets any
# single start fully solve the problem alone and defeats the point of
# multi-start restarts. Capping iterations keeps restarts meaningfully
# different, matching scipy's original multi-start behavior.
_LBFGSB_MAXITER = 20
_LBFGSB_TOL = 1e-6
# ponytail: caps how many distinct training-array lengths' worth of compiled
# JIT closures (#15) are kept alive per (h, loss, kind[, m]) combo. Each
# closure owns its own uncapped internal XLA per-shape cache, so reusing one
# closure across many lengths (e.g. rolling-origin backtests) grew memory
# without bound; bounding the outer cache lets old-length closures get
# evicted and garbage-collected instead. Raise this if a workload legitimately
# cycles through more distinct shapes than it revisits.
_COMPILE_CACHE_SIZE = 16


def _huber(residual: Any, delta: float) -> Array:
    abs_r = jnp.abs(residual)
    quadratic = jnp.minimum(abs_r, delta)
    linear = abs_r - quadratic
    return jnp.mean(0.5 * quadratic**2 + delta * linear)


def _pinball(y_true: Any, y_pred: Any, tau: float) -> Array:
    e = y_true - y_pred
    return jnp.mean(jnp.maximum(tau * e, (tau - 1) * e))


_LOSS_BUILDERS = {
    "mse": lambda delta, tau: lambda yt, yp: jnp.mean((yp - yt) ** 2),
    "mae": lambda delta, tau: lambda yt, yp: jnp.mean(jnp.abs(yp - yt)),
    "huber": lambda delta, tau: lambda yt, yp: _huber(yp - yt, delta),
    "pinball": lambda delta, tau: lambda yt, yp: _pinball(yt, yp, tau),
}


def _loss_key(loss: str | Callable[..., Any], delta: float, tau: float) -> Any:
    """Hashable identity for a loss choice, used as the compile-cache key."""
    if callable(loss):
        return loss
    if loss not in _LOSS_BUILDERS:
        raise ValueError(f"Unknown loss {loss!r}; supported: {list(_LOSS_BUILDERS)}, or a callable")
    if loss == "huber" and delta <= 0:
        raise ValueError(f"huber delta must be positive, got {delta}")
    if loss == "pinball" and not 0 < tau < 1:
        raise ValueError(f"pinball tau must be in (0, 1), got {tau}")
    return (loss, delta, tau)


def _loss_fn_from_key(key: Any) -> Callable[[Any, Any], Any]:
    """(y_true, y_pred) -> scalar, rebuilt from a _loss_key() result."""
    if callable(key):
        return key
    name, delta, tau = key
    return _LOSS_BUILDERS[name](delta, tau)


def _states(
    alpha: Any, beta: Any, phi: Any, level0: Any, trend0: Any, y: Any
) -> tuple[Array, Array]:
    """Level/trend at t=0..T via the (damped) Holt recursion; states[0] is (level0, trend0).

    phi=1.0 recovers plain Holt exactly.
    """

    def step(carry: Any, y_t: Any) -> Any:
        level, trend = carry
        new_level = alpha * y_t + (1 - alpha) * (level + phi * trend)
        new_trend = beta * (new_level - level) + (1 - beta) * phi * trend
        return (new_level, new_trend), (new_level, new_trend)

    _, (levels, trends) = jax.lax.scan(step, (level0, trend0), y)
    return jnp.concatenate([jnp.array([level0]), levels]), jnp.concatenate(
        [jnp.array([trend0]), trends]
    )


def _seasonal_states(
    alpha: Any,
    beta: Any,
    phi: Any,
    gamma: Any,
    level0: Any,
    trend0: Any,
    season0: Any,
    y: Any,
    m: int,
    multiplicative: bool = False,
) -> tuple[Array, Array, Array]:
    """Level/trend/seasonal-buffer at t=0..T via the Holt-Winters recursion.

    season_buf holds the last m seasonal factors, oldest first; season_buf[0]
    is s_{t-m}, the one the level update needs. phi=1.0 recovers plain
    (undamped) Holt-Winters, exactly mirroring _states. multiplicative=True
    swaps the additive (y_t - s_old)/(y_t - new_level) terms for ratios
    (y_t / s_old)/(y_t / new_level); the trend update is unchanged either way.
    """
    season0 = jnp.asarray(season0)

    def step(carry: Any, y_t: Any) -> Any:
        level, trend, season_buf = carry
        s_old = season_buf[0]
        if multiplicative:
            new_level = alpha * (y_t / s_old) + (1 - alpha) * (level + phi * trend)
            new_trend = beta * (new_level - level) + (1 - beta) * phi * trend
            new_season = gamma * (y_t / new_level) + (1 - gamma) * s_old
        else:
            new_level = alpha * (y_t - s_old) + (1 - alpha) * (level + phi * trend)
            new_trend = beta * (new_level - level) + (1 - beta) * phi * trend
            new_season = gamma * (y_t - new_level) + (1 - gamma) * s_old
        new_season_buf = jnp.concatenate([season_buf[1:], jnp.reshape(new_season, (1,))])
        return (new_level, new_trend, new_season_buf), (new_level, new_trend, new_season_buf)

    _, (levels, trends, season_bufs) = jax.lax.scan(step, (level0, trend0, season0), y)
    return (
        jnp.concatenate([jnp.array([level0]), levels]),
        jnp.concatenate([jnp.array([trend0]), trends]),
        jnp.concatenate([season0[None, :], season_bufs], axis=0),
    )


def _UNPACK_UNDAMPED(p: Any) -> tuple[Any, ...]:
    return (p[0], p[1], 1.0, p[2], p[3])


def _UNPACK_DAMPED(p: Any) -> tuple[Any, ...]:
    return (p[0], p[1], p[2], p[3], p[4])


def _UNPACK_SES(p: Any) -> tuple[Any, ...]:
    return (p[0], 0.0, 0.0, p[1], 0.0)


_UNPACK_BY_KIND = {"undamped": _UNPACK_UNDAMPED, "damped": _UNPACK_DAMPED, "ses": _UNPACK_SES}


# Seasonal flat-params layout: box-constrained params first (matching
# _n_trend_params: alpha, beta, [phi,] gamma), then level0, trend0, then the
# m free initial seasonal factors s_1..s_m.
def _UNPACK_SEASONAL(p: Any, m: int) -> tuple[Any, ...]:
    return (p[0], p[1], 1.0, p[2], p[3], p[4], p[5 : 5 + m])


def _UNPACK_DAMPED_SEASONAL(p: Any, m: int) -> tuple[Any, ...]:
    return (p[0], p[1], p[2], p[3], p[4], p[5], p[6 : 6 + m])


# "_mult" kinds share the same flat-params layout/unpack as their additive
# counterpart; only the recursion and combine-op (see _joint_loss_seasonal) differ.
_UNPACK_SEASONAL_BY_KIND = {
    "seasonal": _UNPACK_SEASONAL,
    "damped_seasonal": _UNPACK_DAMPED_SEASONAL,
    "seasonal_mult": _UNPACK_SEASONAL,
    "damped_seasonal_mult": _UNPACK_DAMPED_SEASONAL,
}
_SEASONAL_KINDS = frozenset(_UNPACK_SEASONAL_BY_KIND)
_MULTIPLICATIVE_KINDS = frozenset({"seasonal_mult", "damped_seasonal_mult"})


def _forecast_multiplier(phi: Any, h: int) -> Array:
    """multiplier[k] = sum_{j=1}^{k+1} phi^j; equals arange(1, h+1) when phi == 1."""
    steps = jnp.arange(1, h + 1)
    return jnp.cumsum(phi**steps)


def _forecast_matrix(
    params: Any, y: Any, h: int, unpack: Callable[..., Any]
) -> tuple[Array, Array]:
    """forecasts[t, k] = forecast made at origin t for horizon step k+1, plus actuals[t, k]."""
    alpha, beta, phi, level0, trend0 = unpack(params)
    levels, trends = _states(alpha, beta, phi, level0, trend0, y)
    n_origins = len(y) - h + 1
    multiplier = _forecast_multiplier(phi, h)
    forecasts = levels[:n_origins, None] + multiplier[None, :] * trends[:n_origins, None]
    actuals = jnp.stack([y[t : t + h] for t in range(n_origins)])
    return forecasts, actuals


def _forecast_matrix_seasonal(
    params: Any,
    y: Any,
    h: int,
    unpack: Callable[..., Any],
    m: int,
    multiplicative: bool = False,
) -> tuple[Array, Array]:
    """Mirror _forecast_matrix, but for the seasonal recursion."""
    alpha, beta, phi, gamma, level0, trend0, season0 = unpack(params, m)
    levels, trends, season_bufs = _seasonal_states(
        alpha, beta, phi, gamma, level0, trend0, season0, y, m, multiplicative
    )
    n_origins = len(y) - h + 1
    multiplier = _forecast_multiplier(phi, h)
    trend_part = levels[:n_origins, None] + multiplier[None, :] * trends[:n_origins, None]
    # seasonal forecast for step k (1-indexed) cycles the m most-recent
    # seasonal factors known as of that origin: s_{t+k} == buf_t[(k-1) % m].
    seasonal_idx = jnp.arange(h) % m
    seasonal_part = season_bufs[:n_origins][:, seasonal_idx]
    forecasts = trend_part * seasonal_part if multiplicative else trend_part + seasonal_part
    actuals = jnp.stack([y[t : t + h] for t in range(n_origins)])
    return forecasts, actuals


def _joint_loss(
    params: Any, y: Any, h: int, loss_fn: Callable[..., Any], unpack: Callable[..., Any]
) -> Array:
    forecasts, actuals = _forecast_matrix(params, y, h, unpack)
    return loss_fn(actuals, forecasts)


def _joint_loss_seasonal(
    params: Any,
    y: Any,
    h: int,
    loss_fn: Callable[..., Any],
    unpack: Callable[..., Any],
    m: int,
    multiplicative: bool = False,
) -> Array:
    forecasts, actuals = _forecast_matrix_seasonal(params, y, h, unpack, m, multiplicative)
    return loss_fn(actuals, forecasts)


def _compile_objective_and_grad(
    h: int, loss_key: Any, kind: str, m: int | None = None
) -> tuple[Callable[..., Any], Callable[..., Any]]:
    loss_fn = _loss_fn_from_key(loss_key)

    if kind in _SEASONAL_KINDS:
        assert m is not None  # callers always pass seasonal_period for seasonal kinds
        unpack = _UNPACK_SEASONAL_BY_KIND[kind]
        multiplicative = kind in _MULTIPLICATIVE_KINDS

        def objective(params: Any, y: Any) -> Array:
            return _joint_loss_seasonal(params, y, h, loss_fn, unpack, m, multiplicative)
    else:
        unpack = _UNPACK_BY_KIND[kind]

        def objective(params: Any, y: Any) -> Array:
            return _joint_loss(params, y, h, loss_fn, unpack)

    return jax.jit(objective), jax.jit(jax.grad(objective))


@functools.lru_cache(maxsize=_COMPILE_CACHE_SIZE)
def _compiled_objective_and_grad_cached(
    h: int, loss_key: Any, kind: str, m: int | None, n: int
) -> tuple[Callable[..., Any], Callable[..., Any]]:
    return _compile_objective_and_grad(h, loss_key, kind, m)


def _compiled_objective_and_grad(
    h: int, loss_key: Any, kind: str, m: int | None, n: int
) -> tuple[Callable[..., Any], Callable[..., Any]]:
    """JIT-compiled (objective, grad) for a given (h, loss, kind, m), reused across fits.

    y is a traced arg rather than closed over, so jax.jit's own shape-keyed
    cache is hit whenever a later .fit() call sees a same-length series with
    the same (h, loss, kind, m) instead of recompiling from scratch every call.
    m (seasonal_period) is unused for non-seasonal kinds.

    n (the training array length) is a cache-key-only dimension, not used in
    the body: it forces a fresh, independently-evictable closure per distinct
    length instead of growing one closure's internal per-shape cache forever
    across every length ever seen (#15) -- see _COMPILE_CACHE_SIZE.

    Custom callables bypass the cache: keying on them would pin every
    distinct callable (and its closed-over data) in this module-level cache
    for the life of the process.
    """
    if callable(loss_key):
        return _compile_objective_and_grad(h, loss_key, kind, m)
    return _compiled_objective_and_grad_cached(h, loss_key, kind, m, n)


def _compile_multistart(h: int, loss_key: Any, kind: str) -> Callable[..., Any]:
    """Vmapped L-BFGS-B: runs every restart as one batched solve, not a Python loop.

    Reuses the cached (objective, grad) pair as jaxopt's value_and_grad so the
    autodiff/compile work is identical to before; only the restart loop moves
    from sequential scipy calls into a single jax.vmap'd solver.run.

    Root cause of issue #15: a single call here builds one jax.jit(jax.vmap(...))
    closure that's reused, via the module-level cache below, across every
    later call sharing the same (h, loss, kind) -- including calls with a
    *different* training-array length. Each new length triggers a fresh XLA
    compile that jax caches internally *inside that one closure*, forever,
    with no eviction; a rolling-origin backtest (a different length per fold)
    accumulates one such compile per fold for the life of the process, which
    is exactly the unbounded, roughly-linear memory growth #15 reported (confirmed:
    the real repro, `uv run python -m benchmarks.run_comparison`, OOM-killed
    under a 3G cgroup cap after ~4 minutes). Isolated single-shape .fit() calls
    don't show this, since they only ever produce one compile -- that's why
    they don't reproduce it. Fixed by including the training length in the
    cache key one level up (_compiled_multistart_cached), so each distinct
    length gets its own closure that can be evicted and garbage-collected
    instead of accumulating inside a single unbounded one.
    """
    objective, grad = _compile_objective_and_grad(h, loss_key, kind)
    solver = jaxopt.LBFGSB(
        fun=lambda p, y: (objective(p, y), grad(p, y)),
        value_and_grad=True,
        maxiter=_LBFGSB_MAXITER,
        tol=_LBFGSB_TOL,
    )

    def run_all(x0_batch: Any, lower: Any, upper: Any, y: Any) -> tuple[Any, Any]:
        def run_one(x0: Any) -> Any:
            return solver.run(x0, (lower, upper), y)

        params, state = jax.vmap(run_one)(x0_batch)
        return params, state.value

    return jax.jit(run_all)


@functools.lru_cache(maxsize=_COMPILE_CACHE_SIZE)
def _compiled_multistart_cached(h: int, loss_key: Any, kind: str, n: int) -> Callable[..., Any]:
    return _compile_multistart(h, loss_key, kind)


def _compiled_multistart(h: int, loss_key: Any, kind: str, n: int) -> Callable[..., Any]:
    """JIT-compiled vmapped multi-start solver for (h, loss, kind), reused across fits.

    n (the training array length) is a cache-key-only dimension -- see
    _compiled_objective_and_grad's matching parameter and _COMPILE_CACHE_SIZE.

    Mirrors _compiled_objective_and_grad's cache-bypass for custom callables.
    """
    if callable(loss_key):
        return _compile_multistart(h, loss_key, kind)
    return _compiled_multistart_cached(h, loss_key, kind, n)


def _heuristic_init(y: Any) -> tuple[float, float]:
    k = min(10, len(y))
    trend0, level0 = np.polyfit(np.arange(k), np.asarray(y)[:k], 1)
    return float(level0), float(trend0)


def _heuristic_seasonal_init(y: Any, m: int) -> tuple[float, float, np.ndarray]:
    """Compute level0/trend0 via linear fit on the first ~2 cycles.

    season0 is the per-season-of-cycle average deviation from that line.
    """
    k = min(2 * m, len(y))
    idx = np.arange(k)
    y_k = np.asarray(y)[:k]
    trend0, level0 = np.polyfit(idx, y_k, 1)
    deviations = y_k - (level0 + trend0 * idx)
    season0 = np.array([deviations[idx % m == s].mean() for s in range(m)])
    return float(level0), float(trend0), season0


def _heuristic_seasonal_init_multiplicative(y: Any, m: int) -> tuple[float, float, np.ndarray]:
    """Mirror _heuristic_seasonal_init, but ratio-based.

    season0 is the per-season-of-cycle average of y / trend-line instead of
    y - trend-line.
    """
    k = min(2 * m, len(y))
    idx = np.arange(k)
    y_k = np.asarray(y)[:k]
    trend0, level0 = np.polyfit(idx, y_k, 1)
    ratios = y_k / (level0 + trend0 * idx)
    season0 = np.array([ratios[idx % m == s].mean() for s in range(m)])
    return float(level0), float(trend0), np.maximum(season0, _BOUND_EPS)


def _start_grid(n_starts: int, n_dims: int = 2) -> list[tuple[float, ...]]:
    """n_starts points spanning the full n_dims-cube grid, extremes included."""
    side = int(np.ceil(n_starts ** (1 / n_dims)))
    grid = np.linspace(0.05, 0.95, side)
    combos = list(itertools.product(*([grid] * n_dims)))
    idx = np.unique(np.round(np.linspace(0, len(combos) - 1, n_starts)).astype(int))
    return [combos[i] for i in idx]


class Holt:
    """Holt's linear trend model (undamped, non-seasonal).

    Parameters are fit jointly by minimizing the average forecast error across
    every valid origin and every horizon step 1..h.
    """

    _kind = "undamped"
    _n_trend_params = 2  # alpha, beta
    _n_free_params = 2  # level0, trend0
    seasonal_period: int | None = None  # set by HoltWinters; read here by _resid_std

    def __init__(
        self,
        h: int,
        loss: str | Callable[..., Any] = "mse",
        n_starts: int = 10,
        delta: float = 1.0,
        tau: float = 0.5,
    ) -> None:
        _loss_key(loss, delta, tau)  # validate eagerly, re-resolved fresh in fit()
        self.h = h
        self.loss = loss
        self.n_starts = n_starts
        self.delta = delta
        self.tau = tau
        self.params_: np.ndarray | None = None

    def fit(self, y: Any) -> Self:
        """Fit alpha/beta/level0/trend0 by joint multi-start optimization; returns self."""
        y = np.asarray(y, dtype=float)
        if len(y) < self.h + 1:
            raise ValueError(
                f"Series too short: need at least h + 1 = {self.h + 1} points, got {len(y)}"
            )

        loss_key = _loss_key(self.loss, self.delta, self.tau)
        y_jax = jnp.asarray(y)
        run_all = _compiled_multistart(self.h, loss_key, self._kind, len(y))
        # (level0, trend0) heuristic init; SES._n_free_params == 1 drops trend0.
        free_guess = _heuristic_init(y)[: self._n_free_params]

        starts = np.array(_start_grid(self.n_starts, self._n_trend_params))
        tail = np.tile(free_guess, (len(starts), 1))
        x0_batch = jnp.asarray(np.concatenate([starts, tail], axis=1))
        lower = jnp.array([_BOUND_EPS] * self._n_trend_params + [-jnp.inf] * self._n_free_params)
        upper = jnp.array([1 - _BOUND_EPS] * self._n_trend_params + [jnp.inf] * self._n_free_params)
        if self._kind == "damped":
            upper = upper.at[2].set(_PHI_UPPER)  # phi: cap at the classical damping convention

        params_batch, values = run_all(x0_batch, lower, upper, y_jax)
        self.params_ = np.asarray(params_batch[jnp.argmin(values)])
        self._y = y
        return self

    def predict(self) -> np.ndarray:
        """Forecast the next self.h steps from the fitted level/trend."""
        if self.params_ is None:
            raise ValueError("Model is not fitted yet; call .fit(y) first")
        alpha, beta, level0, trend0 = self.params_
        levels, trends = _states(alpha, beta, 1.0, level0, trend0, self._y)
        level, trend = float(levels[-1]), float(trends[-1])
        return level + np.arange(1, self.h + 1) * trend

    def _resid_std(self) -> np.ndarray:
        """Per-horizon-step std of (actual - forecast) across every valid training origin.

        Recomputes the forecast matrix from the fitted params, dispatching
        seasonal vs. non-seasonal by _kind exactly like
        _compile_objective_and_grad does. Not cached: it's one cheap forward
        pass, and caching risks staleness across refits.

        With very short series (few valid origins) this degenerates, e.g. a
        single origin gives 0 std and an all-zero-width interval -- accepted
        as a known limitation rather than guarded against here.
        """
        y_jax = jnp.asarray(self._y)
        if self._kind in _SEASONAL_KINDS:
            assert self.seasonal_period is not None  # set by HoltWinters for seasonal kinds
            unpack = _UNPACK_SEASONAL_BY_KIND[self._kind]
            multiplicative = self._kind in _MULTIPLICATIVE_KINDS
            forecasts, actuals = _forecast_matrix_seasonal(
                self.params_, y_jax, self.h, unpack, self.seasonal_period, multiplicative
            )
        else:
            unpack = _UNPACK_BY_KIND[self._kind]
            forecasts, actuals = _forecast_matrix(self.params_, y_jax, self.h, unpack)
        return np.asarray(jnp.std(actuals - forecasts, axis=0))

    def predict_interval(
        self, coverage: float | list[float] | tuple[float, ...] = 0.9
    ) -> tuple[np.ndarray, np.ndarray] | dict[float, tuple[np.ndarray, np.ndarray]]:
        """Empirical per-horizon-step Gaussian interval(s) around .predict().

        coverage: a single float -> (lower, upper) array pair; a list of
        floats -> {coverage: (lower, upper)} dict.
        """
        if self.params_ is None:
            raise ValueError("Model is not fitted yet; call .fit(y) first")
        point = self.predict()
        std = self._resid_std()

        def bounds(c: float) -> tuple[np.ndarray, np.ndarray]:
            z = norm.ppf(0.5 + c / 2)
            return point - z * std, point + z * std

        if isinstance(coverage, (list, tuple)):
            return {c: bounds(c) for c in coverage}
        return bounds(coverage)


class SES(Holt):
    """Simple exponential smoothing: a flat (no trend) forecast.

    alpha and level0 are fit jointly against the same joint h-step loss as
    Holt; beta/phi/trend0 are pinned to 0 so the recursion collapses to plain
    SES and every forecast horizon step returns the same level.
    """

    _kind = "ses"
    _n_trend_params = 1  # alpha
    _n_free_params = 1  # level0

    def predict(self) -> np.ndarray:
        """Forecast the next self.h steps as a flat repeat of the fitted level."""
        if self.params_ is None:
            raise ValueError("Model is not fitted yet; call .fit(y) first")
        alpha, level0 = self.params_
        levels, _ = _states(alpha, 0.0, 0.0, level0, 0.0, self._y)
        return np.full(self.h, float(levels[-1]))


class DampedHolt(Holt):
    """Holt's linear trend model with a damped trend (non-seasonal).

    Adds a box-constrained damping parameter phi in (0, 1), jointly optimized
    alongside alpha/beta against the same joint h-step loss as Holt.
    """

    _kind = "damped"
    _n_trend_params = 3  # alpha, beta, phi

    def predict(self) -> np.ndarray:
        """Forecast the next self.h steps from the fitted level/trend/damping."""
        if self.params_ is None:
            raise ValueError("Model is not fitted yet; call .fit(y) first")
        alpha, beta, phi, level0, trend0 = self.params_
        levels, trends = _states(alpha, beta, phi, level0, trend0, self._y)
        level, trend = float(levels[-1]), float(trends[-1])
        multiplier = np.asarray(_forecast_multiplier(phi, self.h))
        return level + multiplier * trend


class HoltWinters(Holt):
    """Holt-Winters: Holt's linear trend plus additive or multiplicative seasonality.

    Seasonal factors s_1..s_m (the initial cycle) are free parameters
    (box-constrained to stay positive when seasonal_type="multiplicative"),
    jointly optimized alongside alpha/beta/gamma/level0/trend0 against the
    same joint h-step loss as Holt. phi is fixed at 1.0 (undamped trend).
    """

    _kind = "seasonal"
    _n_trend_params = 3  # alpha, beta, gamma
    seasonal_period: int  # narrows Holt's int | None: always set by this subclass's __init__

    def __init__(
        self,
        h: int,
        seasonal_period: int,
        seasonal_type: str = "additive",
        loss: str | Callable[..., Any] = "mse",
        n_starts: int = 10,
        delta: float = 1.0,
        tau: float = 0.5,
    ) -> None:
        super().__init__(h, loss=loss, n_starts=n_starts, delta=delta, tau=tau)
        if seasonal_type not in ("additive", "multiplicative"):
            raise ValueError(
                f"Unknown seasonal_type {seasonal_type!r}; supported: 'additive', 'multiplicative'"
            )
        self.seasonal_period = seasonal_period
        self.seasonal_type = seasonal_type
        if seasonal_type == "multiplicative":
            self._kind = self._kind + "_mult"

    def fit(self, y: Any) -> Self:
        """Fit trend and seasonal params by joint multi-start optimization; returns self."""
        y = np.asarray(y, dtype=float)
        m = self.seasonal_period
        multiplicative = self.seasonal_type == "multiplicative"
        if len(y) < self.h + 1:
            raise ValueError(
                f"Series too short: need at least h + 1 = {self.h + 1} points, got {len(y)}"
            )
        if len(y) < 2 * m:
            raise ValueError(
                "Series too short: need at least 2 * seasonal_period = "
                f"{2 * m} points, got {len(y)}"
            )

        loss_key = _loss_key(self.loss, self.delta, self.tau)
        y_jax = jnp.asarray(y)
        objective, grad = _compiled_objective_and_grad(self.h, loss_key, self._kind, m, len(y))
        if multiplicative:
            level0_guess, trend0_guess, season0_guess = _heuristic_seasonal_init_multiplicative(
                y, m
            )
            season_bounds = [(_BOUND_EPS, None)] * m
        else:
            level0_guess, trend0_guess, season0_guess = _heuristic_seasonal_init(y, m)
            season_bounds = [(None, None)] * m
        bounds = (
            [(_BOUND_EPS, 1 - _BOUND_EPS)] * self._n_trend_params
            + [(None, None), (None, None)]
            + season_bounds
        )
        if self._kind.startswith("damped"):
            bounds[2] = (_BOUND_EPS, _PHI_UPPER)  # phi: cap at the classical damping convention

        best = None
        for start in _start_grid(self.n_starts, self._n_trend_params):
            x0 = np.concatenate([start, [level0_guess, trend0_guess], season0_guess])
            result = minimize(
                lambda p: float(objective(p, y_jax)),
                x0,
                jac=lambda p: np.asarray(grad(p, y_jax)),
                method="L-BFGS-B",
                bounds=bounds,
            )
            if best is None or result.fun < best.fun:
                best = result

        assert best is not None  # _start_grid always yields at least one start
        self.params_ = best.x
        self._y = y
        self._rescale_seasonal()
        return self

    def _rescale_seasonal(self) -> None:
        """Rescale seasonal factors to a canonical mean, without changing any forecast.

        Additive: shift seasonal factors to mean 0, folding the removed
        mean into level0. Both updates are convex combinations of (y_t - s)
        and (level + s), so a uniform +c/-c shift cancels through the whole
        recursion and leaves every forecast unchanged.

        Multiplicative: scale seasonal factors to mean 1, folding the removed
        scale into level0 and trend0. Scaling season by 1/c and (level, trend)
        by c leaves every one of level_t, trend_t, season_t scaled by the same
        c (by induction: level_t = alpha*(y_t/s_old) + (1-alpha)*(level+phi*trend)
        scales by c when s_old scales by 1/c and (level, trend) scale by c;
        trend_t is a difference of levels so it scales by c too; season_t =
        gamma*(y_t/level_t) + (1-gamma)*s_old scales by 1/c by the same
        argument). Forecasts are (level + multiplier*trend) * seasonal, i.e.
        c * (1/c) = 1, so they're unchanged.
        """
        assert self.params_ is not None  # only called from fit(), right after it's set
        m = self.seasonal_period
        n = self._n_trend_params
        season0 = self.params_[n + 2 : n + 2 + m]
        c = season0.mean()
        if self.seasonal_type == "multiplicative":
            self.params_[n] *= c
            self.params_[n + 1] *= c
            self.params_[n + 2 : n + 2 + m] /= c
        else:
            self.params_[n] += c
            self.params_[n + 2 : n + 2 + m] -= c

    def predict(self) -> np.ndarray:
        """Forecast the next self.h steps from the fitted level/trend/seasonal factors."""
        if self.params_ is None:
            raise ValueError("Model is not fitted yet; call .fit(y) first")
        m = self.seasonal_period
        multiplicative = self.seasonal_type == "multiplicative"
        alpha, beta, phi, gamma, level0, trend0, season0 = _UNPACK_SEASONAL_BY_KIND[self._kind](
            self.params_, m
        )
        levels, trends, season_bufs = _seasonal_states(
            alpha, beta, phi, gamma, level0, trend0, season0, self._y, m, multiplicative
        )
        level, trend = float(levels[-1]), float(trends[-1])
        multiplier = np.asarray(_forecast_multiplier(phi, self.h))
        seasonal = np.asarray(season_bufs[-1])[np.arange(self.h) % m]
        trend_part = level + multiplier * trend
        return trend_part * seasonal if multiplicative else trend_part + seasonal


class DampedHoltWinters(HoltWinters):
    """Holt-Winters with additive or multiplicative seasonality and a damped trend."""

    _kind = "damped_seasonal"
    _n_trend_params = 4  # alpha, beta, phi, gamma
