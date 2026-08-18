"""Holt's linear trend model, fit by minimizing a joint h-step-ahead loss."""

import functools
import itertools

import jax
import jax.numpy as jnp
import jaxopt
import numpy as np

jax.config.update("jax_enable_x64", True)

_BOUND_EPS = 1e-4
# L-BFGS-B stopping criteria for the vmapped multi-start solver below. A low
# maxiter is deliberate, not just a speed knob: jaxopt's LBFGSB converges
# more thoroughly per-start than scipy's did, so a generous budget lets any
# single start fully solve the problem alone and defeats the point of
# multi-start restarts. Capping iterations keeps restarts meaningfully
# different, matching scipy's original multi-start behavior.
_LBFGSB_MAXITER = 20
_LBFGSB_TOL = 1e-6


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


def _seasonal_states(alpha, beta, phi, gamma, level0, trend0, season0, y, m):
    """Level/trend/seasonal-buffer at t=0..T via the additive Holt-Winters recursion.

    season_buf holds the last m seasonal factors, oldest first; season_buf[0]
    is s_{t-m}, the one the level update needs. phi=1.0 recovers plain
    (undamped) Holt-Winters, exactly mirroring _states.
    """
    season0 = jnp.asarray(season0)

    def step(carry, y_t):
        level, trend, season_buf = carry
        s_old = season_buf[0]
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


_UNPACK_UNDAMPED = lambda p: (p[0], p[1], 1.0, p[2], p[3])
_UNPACK_DAMPED = lambda p: (p[0], p[1], p[2], p[3], p[4])
_UNPACK_BY_KIND = {"undamped": _UNPACK_UNDAMPED, "damped": _UNPACK_DAMPED}

# Seasonal flat-params layout: box-constrained params first (matching
# _n_trend_params: alpha, beta, [phi,] gamma), then level0, trend0, then the
# m free initial seasonal factors s_1..s_m.
_UNPACK_SEASONAL = lambda p, m: (p[0], p[1], 1.0, p[2], p[3], p[4], p[5 : 5 + m])
_UNPACK_DAMPED_SEASONAL = lambda p, m: (p[0], p[1], p[2], p[3], p[4], p[5], p[6 : 6 + m])
_UNPACK_SEASONAL_BY_KIND = {"seasonal": _UNPACK_SEASONAL, "damped_seasonal": _UNPACK_DAMPED_SEASONAL}
_SEASONAL_KINDS = frozenset(_UNPACK_SEASONAL_BY_KIND)


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


def _joint_loss_seasonal(params, y, h, loss_fn, unpack, m):
    alpha, beta, phi, gamma, level0, trend0, season0 = unpack(params, m)
    levels, trends, season_bufs = _seasonal_states(
        alpha, beta, phi, gamma, level0, trend0, season0, y, m
    )
    n_origins = len(y) - h + 1
    multiplier = _forecast_multiplier(phi, h)
    trend_part = levels[:n_origins, None] + multiplier[None, :] * trends[:n_origins, None]
    # seasonal forecast for step k (1-indexed) cycles the m most-recent
    # seasonal factors known as of that origin: s_{t+k} == buf_t[(k-1) % m].
    seasonal_idx = jnp.arange(h) % m
    seasonal_part = season_bufs[:n_origins][:, seasonal_idx]
    forecasts = trend_part + seasonal_part
    actuals = jnp.stack([y[t : t + h] for t in range(n_origins)])
    return loss_fn(actuals, forecasts)


def _compile_objective_and_grad(h, loss_key, kind, m=None):
    loss_fn = _loss_fn_from_key(loss_key)

    if kind in _SEASONAL_KINDS:
        unpack = _UNPACK_SEASONAL_BY_KIND[kind]

        def objective(params, y):
            return _joint_loss_seasonal(params, y, h, loss_fn, unpack, m)
    else:
        unpack = _UNPACK_BY_KIND[kind]

        def objective(params, y):
            return _joint_loss(params, y, h, loss_fn, unpack)

    return jax.jit(objective), jax.jit(jax.grad(objective))


@functools.lru_cache(maxsize=None)
def _compiled_objective_and_grad_cached(h, loss_key, kind, m=None):
    return _compile_objective_and_grad(h, loss_key, kind, m)


def _compiled_objective_and_grad(h, loss_key, kind, m=None):
    """JIT-compiled (objective, grad) for a given (h, loss, kind, m), reused across fits.

    y is a traced arg rather than closed over, so jax.jit's own shape-keyed
    cache is hit whenever a later .fit() call sees a same-length series with
    the same (h, loss, kind, m) instead of recompiling from scratch every call.
    m (seasonal_period) is unused for non-seasonal kinds.

    Custom callables bypass the cache: keying on them would pin every
    distinct callable (and its closed-over data) in this module-level cache
    for the life of the process.
    """
    if callable(loss_key):
        return _compile_objective_and_grad(h, loss_key, kind, m)
    return _compiled_objective_and_grad_cached(h, loss_key, kind, m)


def _compile_multistart(h, loss_key, kind):
    """Vmapped L-BFGS-B: runs every restart as one batched solve, not a Python loop.

    Reuses the cached (objective, grad) pair as jaxopt's value_and_grad so the
    autodiff/compile work is identical to before; only the restart loop moves
    from sequential scipy calls into a single jax.vmap'd solver.run.
    """
    objective, grad = _compile_objective_and_grad(h, loss_key, kind)
    solver = jaxopt.LBFGSB(
        fun=lambda p, y: (objective(p, y), grad(p, y)),
        value_and_grad=True,
        maxiter=_LBFGSB_MAXITER,
        tol=_LBFGSB_TOL,
    )

    def run_all(x0_batch, lower, upper, y):
        run_one = lambda x0: solver.run(x0, (lower, upper), y)
        params, state = jax.vmap(run_one)(x0_batch)
        return params, state.value

    return jax.jit(run_all)


@functools.lru_cache(maxsize=None)
def _compiled_multistart_cached(h, loss_key, kind):
    return _compile_multistart(h, loss_key, kind)


def _compiled_multistart(h, loss_key, kind):
    """JIT-compiled vmapped multi-start solver for (h, loss, kind), reused across fits.

    Mirrors _compiled_objective_and_grad's cache-bypass for custom callables.
    """
    if callable(loss_key):
        return _compile_multistart(h, loss_key, kind)
    return _compiled_multistart_cached(h, loss_key, kind)


def _heuristic_init(y):
    k = min(10, len(y))
    trend0, level0 = np.polyfit(np.arange(k), np.asarray(y)[:k], 1)
    return float(level0), float(trend0)


def _heuristic_seasonal_init(y, m):
    """level0/trend0 via linear fit on the first ~2 cycles; season0 as the
    per-season-of-cycle average deviation from that line."""
    k = min(2 * m, len(y))
    idx = np.arange(k)
    y_k = np.asarray(y)[:k]
    trend0, level0 = np.polyfit(idx, y_k, 1)
    deviations = y_k - (level0 + trend0 * idx)
    season0 = np.array([deviations[idx % m == s].mean() for s in range(m)])
    return float(level0), float(trend0), season0


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
        run_all = _compiled_multistart(self.h, loss_key, self._kind)
        level0_guess, trend0_guess = _heuristic_init(y)

        starts = np.array(_start_grid(self.n_starts, self._n_trend_params))
        tail = np.tile([level0_guess, trend0_guess], (len(starts), 1))
        x0_batch = jnp.asarray(np.concatenate([starts, tail], axis=1))
        lower = jnp.array([_BOUND_EPS] * self._n_trend_params + [-jnp.inf, -jnp.inf])
        upper = jnp.array([1 - _BOUND_EPS] * self._n_trend_params + [jnp.inf, jnp.inf])

        params_batch, values = run_all(x0_batch, lower, upper, y_jax)
        self.params_ = np.asarray(params_batch[jnp.argmin(values)])
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


class HoltWinters(Holt):
    """Holt-Winters: Holt's linear trend plus additive seasonality.

    Seasonal factors s_1..s_m (the initial cycle) are free parameters,
    jointly optimized alongside alpha/beta/gamma/level0/trend0 against the
    same joint h-step loss as Holt. phi is fixed at 1.0 (undamped trend).
    """

    _kind = "seasonal"
    _n_trend_params = 3  # alpha, beta, gamma

    def __init__(self, h, seasonal_period, loss="mse", n_starts=10, delta=1.0, tau=0.5):
        super().__init__(h, loss=loss, n_starts=n_starts, delta=delta, tau=tau)
        self.seasonal_period = seasonal_period

    def fit(self, y):
        y = np.asarray(y, dtype=float)
        m = self.seasonal_period
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
        objective, grad = _compiled_objective_and_grad(self.h, loss_key, self._kind, m)
        level0_guess, trend0_guess, season0_guess = _heuristic_seasonal_init(y, m)
        bounds = [(_BOUND_EPS, 1 - _BOUND_EPS)] * self._n_trend_params + [
            (None, None)
        ] * (2 + m)

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

        self.params_ = best.x
        self._y = y
        self._rescale_seasonal()
        return self

    def _rescale_seasonal(self):
        """Shift seasonal factors to mean 0, folding the removed mean into
        level0. Both updates are convex combinations of (y_t - s) and
        (level + s), so a uniform +c/-c shift cancels through the whole
        recursion and leaves every forecast unchanged."""
        m = self.seasonal_period
        n = self._n_trend_params
        season0 = self.params_[n + 2 : n + 2 + m]
        c = season0.mean()
        self.params_[n] += c
        self.params_[n + 2 : n + 2 + m] -= c

    def predict(self):
        if self.params_ is None:
            raise ValueError("Model is not fitted yet; call .fit(y) first")
        m = self.seasonal_period
        alpha, beta, phi, gamma, level0, trend0, season0 = _UNPACK_SEASONAL_BY_KIND[
            self._kind
        ](self.params_, m)
        levels, trends, season_bufs = _seasonal_states(
            alpha, beta, phi, gamma, level0, trend0, season0, self._y, m
        )
        level, trend = float(levels[-1]), float(trends[-1])
        multiplier = np.asarray(_forecast_multiplier(phi, self.h))
        seasonal = np.asarray(season_bufs[-1])[np.arange(self.h) % m]
        return level + multiplier * trend + seasonal


class DampedHoltWinters(HoltWinters):
    """Holt-Winters with additive seasonality and a damped trend."""

    _kind = "damped_seasonal"
    _n_trend_params = 4  # alpha, beta, phi, gamma
