"""Holt's linear trend model, fit by minimizing a joint h-step-ahead loss."""

import functools
import itertools

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize

jax.config.update("jax_enable_x64", True)

_BOUND_EPS = 1e-4


def _huber(residual, delta):
    abs_r = jnp.abs(residual)
    quadratic = jnp.minimum(abs_r, delta)
    linear = abs_r - quadratic
    return jnp.mean(0.5 * quadratic**2 + delta * linear)


def _pinball(y_true, y_pred, tau):
    e = y_true - y_pred
    return jnp.mean(jnp.maximum(tau * e, (tau - 1) * e))


_LOSS_BUILDERS = {
    "mse": lambda delta, tau: (lambda yt, yp: jnp.mean((yp - yt) ** 2)),
    "mae": lambda delta, tau: (lambda yt, yp: jnp.mean(jnp.abs(yp - yt))),
    "huber": lambda delta, tau: (lambda yt, yp: _huber(yp - yt, delta)),
    "pinball": lambda delta, tau: (lambda yt, yp: _pinball(yt, yp, tau)),
}


def _loss_key(loss, delta, tau):
    """Hashable identity for a loss choice, used as the compile-cache key."""
    if callable(loss):
        return loss
    if loss not in _LOSS_BUILDERS:
        raise ValueError(
            f"Unknown loss {loss!r}; supported: {list(_LOSS_BUILDERS)}, or a callable"
        )
    if loss == "huber" and delta <= 0:
        raise ValueError(f"huber delta must be positive, got {delta}")
    if loss == "pinball" and not 0 < tau < 1:
        raise ValueError(f"pinball tau must be in (0, 1), got {tau}")
    return (loss, delta, tau)


def _loss_fn_from_key(key):
    """(y_true, y_pred) -> scalar, rebuilt from a _loss_key() result."""
    if callable(key):
        return key
    name, delta, tau = key
    return _LOSS_BUILDERS[name](delta, tau)


def _states(alpha, beta, phi, level0, trend0, y):
    """Level/trend at t=0..T via the (damped) Holt recursion; states[0] is (level0, trend0).

    phi=1.0 recovers plain Holt exactly.
    """

    def step(carry, y_t):
        level, trend = carry
        new_level = alpha * y_t + (1 - alpha) * (level + phi * trend)
        new_trend = beta * (new_level - level) + (1 - beta) * phi * trend
        return (new_level, new_trend), (new_level, new_trend)

    _, (levels, trends) = jax.lax.scan(step, (level0, trend0), y)
    return jnp.concatenate([jnp.array([level0]), levels]), jnp.concatenate(
        [jnp.array([trend0]), trends]
    )


_UNPACK_UNDAMPED = lambda p: (p[0], p[1], 1.0, p[2], p[3])
_UNPACK_DAMPED = lambda p: (p[0], p[1], p[2], p[3], p[4])
_UNPACK_BY_KIND = {"undamped": _UNPACK_UNDAMPED, "damped": _UNPACK_DAMPED}


def _forecast_multiplier(phi, h):
    """multiplier[k] = sum_{j=1}^{k+1} phi^j; equals arange(1, h+1) when phi == 1."""
    steps = jnp.arange(1, h + 1)
    return jnp.cumsum(phi**steps)


def _joint_loss(params, y, h, loss_fn, unpack):
    alpha, beta, phi, level0, trend0 = unpack(params)
    levels, trends = _states(alpha, beta, phi, level0, trend0, y)
    n_origins = len(y) - h + 1
    multiplier = _forecast_multiplier(phi, h)
    # forecasts[t, k] = forecast made at origin t for horizon step k+1
    forecasts = levels[:n_origins, None] + multiplier[None, :] * trends[:n_origins, None]
    actuals = jnp.stack([y[t : t + h] for t in range(n_origins)])
    return loss_fn(actuals, forecasts)


def _compile_objective_and_grad(h, loss_key, kind):
    loss_fn = _loss_fn_from_key(loss_key)
    unpack = _UNPACK_BY_KIND[kind]

    def objective(params, y):
        return _joint_loss(params, y, h, loss_fn, unpack)

    return jax.jit(objective), jax.jit(jax.grad(objective))


@functools.lru_cache(maxsize=None)
def _compiled_objective_and_grad_cached(h, loss_key, kind):
    return _compile_objective_and_grad(h, loss_key, kind)


def _compiled_objective_and_grad(h, loss_key, kind):
    """JIT-compiled (objective, grad) for a given (h, loss, kind), reused across fits.

    y is a traced arg rather than closed over, so jax.jit's own shape-keyed
    cache is hit whenever a later .fit() call sees a same-length series with
    the same (h, loss, kind) instead of recompiling from scratch every call.

    Custom callables bypass the cache: keying on them would pin every
    distinct callable (and its closed-over data) in this module-level cache
    for the life of the process.
    """
    if callable(loss_key):
        return _compile_objective_and_grad(h, loss_key, kind)
    return _compiled_objective_and_grad_cached(h, loss_key, kind)


def _heuristic_init(y):
    k = min(10, len(y))
    trend0, level0 = np.polyfit(np.arange(k), np.asarray(y)[:k], 1)
    return float(level0), float(trend0)


def _start_grid(n_starts, n_dims=2):
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

    def __init__(self, h, loss="mse", n_starts=10, delta=1.0, tau=0.5):
        _loss_key(loss, delta, tau)  # validate eagerly, re-resolved fresh in fit()
        self.h = h
        self.loss = loss
        self.n_starts = n_starts
        self.delta = delta
        self.tau = tau
        self.params_ = None

    def fit(self, y):
        y = np.asarray(y, dtype=float)
        if len(y) < self.h + 1:
            raise ValueError(
                f"Series too short: need at least h + 1 = {self.h + 1} points, got {len(y)}"
            )

        loss_key = _loss_key(self.loss, self.delta, self.tau)
        y_jax = jnp.asarray(y)
        objective, grad = _compiled_objective_and_grad(self.h, loss_key, self._kind)
        level0_guess, trend0_guess = _heuristic_init(y)
        bounds = [(_BOUND_EPS, 1 - _BOUND_EPS)] * self._n_trend_params + [
            (None, None),
            (None, None),
        ]

        best = None
        for start in _start_grid(self.n_starts, self._n_trend_params):
            x0 = np.array([*start, level0_guess, trend0_guess])
            result = minimize(
                lambda p: float(objective(p, y_jax)),
                x0,
                jac=lambda p: np.asarray(grad(p, y_jax)),
                method="L-BFGS-B",
                bounds=bounds,
            )
            if best is None or result.fun < best.fun:
                best = result

        self.params_ = best.x
        self._y = y
        return self

    def predict(self):
        if self.params_ is None:
            raise ValueError("Model is not fitted yet; call .fit(y) first")
        alpha, beta, level0, trend0 = self.params_
        levels, trends = _states(alpha, beta, 1.0, level0, trend0, self._y)
        level, trend = float(levels[-1]), float(trends[-1])
        return level + np.arange(1, self.h + 1) * trend


class DampedHolt(Holt):
    """Holt's linear trend model with a damped trend (non-seasonal).

    Adds a box-constrained damping parameter phi in (0, 1), jointly optimized
    alongside alpha/beta against the same joint h-step loss as Holt.
    """

    _kind = "damped"
    _n_trend_params = 3  # alpha, beta, phi

    def predict(self):
        if self.params_ is None:
            raise ValueError("Model is not fitted yet; call .fit(y) first")
        alpha, beta, phi, level0, trend0 = self.params_
        levels, trends = _states(alpha, beta, phi, level0, trend0, self._y)
        level, trend = float(levels[-1]), float(trends[-1])
        multiplier = np.asarray(_forecast_multiplier(phi, self.h))
        return level + multiplier * trend
