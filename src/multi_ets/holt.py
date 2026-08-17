"""Holt's linear trend model, fit by minimizing a joint h-step-ahead loss."""

import itertools

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize

jax.config.update("jax_enable_x64", True)

_BOUND_EPS = 1e-4
_PARAM_BOUNDS = [(_BOUND_EPS, 1 - _BOUND_EPS)] * 2  # alpha, beta

_LOSSES = {
    "mse": lambda err: jnp.mean(err**2),
}


def _states(alpha, beta, level0, trend0, y):
    """Level/trend at t=0..T via the Holt recursion; states[0] is (level0, trend0)."""

    def step(carry, y_t):
        level, trend = carry
        new_level = alpha * y_t + (1 - alpha) * (level + trend)
        new_trend = beta * (new_level - level) + (1 - beta) * trend
        return (new_level, new_trend), (new_level, new_trend)

    _, (levels, trends) = jax.lax.scan(step, (level0, trend0), y)
    return jnp.concatenate([jnp.array([level0]), levels]), jnp.concatenate(
        [jnp.array([trend0]), trends]
    )


def _joint_loss(params, y, h, loss_fn):
    alpha, beta, level0, trend0 = params
    levels, trends = _states(alpha, beta, level0, trend0, y)
    n_origins = len(y) - h + 1
    steps = jnp.arange(1, h + 1)
    # forecasts[t, k] = forecast made at origin t for horizon step k+1
    forecasts = levels[:n_origins, None] + steps[None, :] * trends[:n_origins, None]
    actuals = jnp.stack([y[t : t + h] for t in range(n_origins)])
    return loss_fn(forecasts - actuals)


def _heuristic_init(y):
    k = min(10, len(y))
    trend0, level0 = np.polyfit(np.arange(k), np.asarray(y)[:k], 1)
    return float(level0), float(trend0)


def _start_grid(n_starts):
    """n_starts points spanning the full (alpha, beta) grid, extremes included."""
    side = int(np.ceil(np.sqrt(n_starts)))
    grid = np.linspace(0.05, 0.95, side)
    combos = list(itertools.product(grid, grid))
    idx = np.unique(np.round(np.linspace(0, len(combos) - 1, n_starts)).astype(int))
    return [combos[i] for i in idx]


class Holt:
    """Holt's linear trend model (undamped, non-seasonal).

    Parameters are fit jointly by minimizing the average forecast error across
    every valid origin and every horizon step 1..h.
    """

    def __init__(self, h, loss="mse", n_starts=10):
        if loss not in _LOSSES:
            raise ValueError(f"Unknown loss {loss!r}; supported: {list(_LOSSES)}")
        self.h = h
        self.loss = loss
        self.n_starts = n_starts
        self.params_ = None

    def fit(self, y):
        y = np.asarray(y, dtype=float)
        if len(y) < self.h + 1:
            raise ValueError(
                f"Series too short: need at least h + 1 = {self.h + 1} points, got {len(y)}"
            )

        loss_fn = _LOSSES[self.loss]
        y_jax = jnp.asarray(y)
        objective = jax.jit(lambda p: _joint_loss(p, y_jax, self.h, loss_fn))
        grad = jax.jit(jax.grad(objective))
        level0_guess, trend0_guess = _heuristic_init(y)

        best = None
        for alpha0, beta0 in _start_grid(self.n_starts):
            x0 = np.array([alpha0, beta0, level0_guess, trend0_guess])
            result = minimize(
                lambda p: float(objective(p)),
                x0,
                jac=lambda p: np.asarray(grad(p)),
                method="L-BFGS-B",
                bounds=_PARAM_BOUNDS + [(None, None), (None, None)],
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
        levels, trends = _states(alpha, beta, level0, trend0, self._y)
        level, trend = float(levels[-1]), float(trends[-1])
        return level + np.arange(1, self.h + 1) * trend
