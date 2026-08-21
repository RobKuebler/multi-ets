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


def _masked_mean(x: Any, mask: Any | None) -> Array:
    """jnp.mean(x), or a mean over only the rows where mask is True.

    mask is None means "no padding" (the plain mean). Otherwise mask is a
    per-origin (axis-0) boolean array; masked-out rows are zeroed before
    summing, which is exact (not approximate) because summing extra 0.0
    terms never perturbs an IEEE-754 float sum -- so this reproduces
    jnp.mean over just the valid rows bit-for-bit (see #17 / _pad_to_bucket).
    """
    if mask is None:
        return jnp.mean(x)
    weighted = jnp.where(mask[:, None], x, 0.0)
    return jnp.sum(weighted) / (jnp.sum(mask) * x.shape[-1])


def _huber(residual: Any, delta: float, mask: Any | None = None) -> Array:
    abs_r = jnp.abs(residual)
    quadratic = jnp.minimum(abs_r, delta)
    linear = abs_r - quadratic
    return _masked_mean(0.5 * quadratic**2 + delta * linear, mask)


def _pinball(y_true: Any, y_pred: Any, tau: float, mask: Any | None = None) -> Array:
    e = y_true - y_pred
    return _masked_mean(jnp.maximum(tau * e, (tau - 1) * e), mask)


_LOSS_BUILDERS = {
    "mse": lambda delta, tau: lambda yt, yp, mask=None: _masked_mean((yp - yt) ** 2, mask),
    "mae": lambda delta, tau: lambda yt, yp, mask=None: _masked_mean(jnp.abs(yp - yt), mask),
    "huber": lambda delta, tau: lambda yt, yp, mask=None: _huber(yp - yt, delta, mask),
    "pinball": lambda delta, tau: lambda yt, yp, mask=None: _pinball(yt, yp, tau, mask),
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


def _origin_mask(n_origins: int, n_valid: Any | None, h: int) -> Any | None:
    """None (no padding), or a bool[n_origins] marking origins backed by real (unpadded) data."""
    if n_valid is None:
        return None
    return jnp.arange(n_origins) < (n_valid - h + 1)


def _forecast_matrix(
    params: Any, y: Any, h: int, unpack: Callable[..., Any], n_valid: Any | None = None
) -> tuple[Array, Array, Array | None]:
    """forecasts[t, k] = forecast made at origin t for horizon step k+1, plus actuals[t, k].

    n_valid, if given, is the real (pre-padding) series length; the returned
    mask marks which origin rows are backed by real data vs. right-padding
    (see _pad_to_bucket) and is None when n_valid is None (no padding, i.e.
    the previous unmasked behavior).
    """
    alpha, beta, phi, level0, trend0 = unpack(params)
    levels, trends = _states(alpha, beta, phi, level0, trend0, y)
    n_origins = len(y) - h + 1
    multiplier = _forecast_multiplier(phi, h)
    forecasts = levels[:n_origins, None] + multiplier[None, :] * trends[:n_origins, None]
    actuals = jnp.stack([y[t : t + h] for t in range(n_origins)])
    return forecasts, actuals, _origin_mask(n_origins, n_valid, h)


def _forecast_matrix_seasonal(
    params: Any,
    y: Any,
    h: int,
    unpack: Callable[..., Any],
    m: int,
    multiplicative: bool = False,
    n_valid: Any | None = None,
) -> tuple[Array, Array, Array | None]:
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
    return forecasts, actuals, _origin_mask(n_origins, n_valid, h)


def _joint_loss(
    params: Any,
    y: Any,
    h: int,
    loss_fn: Callable[..., Any],
    unpack: Callable[..., Any],
    n_valid: Any | None = None,
) -> Array:
    forecasts, actuals, mask = _forecast_matrix(params, y, h, unpack, n_valid)
    return loss_fn(actuals, forecasts) if mask is None else loss_fn(actuals, forecasts, mask)


def _joint_loss_seasonal(
    params: Any,
    y: Any,
    h: int,
    loss_fn: Callable[..., Any],
    unpack: Callable[..., Any],
    m: int,
    multiplicative: bool = False,
    n_valid: Any | None = None,
) -> Array:
    forecasts, actuals, mask = _forecast_matrix_seasonal(
        params, y, h, unpack, m, multiplicative, n_valid
    )
    return loss_fn(actuals, forecasts) if mask is None else loss_fn(actuals, forecasts, mask)


def _compile_objective_and_grad(
    h: int, loss_key: Any, kind: str, m: int | None = None
) -> tuple[Callable[..., Any], Callable[..., Any]]:
    """Build (objective, grad), both taking (params, y, n_valid).

    n_valid is always accepted for a uniform call signature, but is only
    forwarded to the padding/masking scheme in _pad_to_bucket for built-in
    (str/tuple-keyed) losses; custom callables ignore it, since their loss_fn
    keeps the user-facing 2-arg (y_true, y_pred) signature untouched.
    """
    loss_fn = _loss_fn_from_key(loss_key)
    custom = callable(loss_key)

    if kind in _SEASONAL_KINDS:
        assert m is not None  # callers always pass seasonal_period for seasonal kinds
        unpack = _UNPACK_SEASONAL_BY_KIND[kind]
        multiplicative = kind in _MULTIPLICATIVE_KINDS

        def objective(params: Any, y: Any, n_valid: Any) -> Array:
            return _joint_loss_seasonal(
                params, y, h, loss_fn, unpack, m, multiplicative, None if custom else n_valid
            )
    else:
        unpack = _UNPACK_BY_KIND[kind]

        def objective(params: Any, y: Any, n_valid: Any) -> Array:
            return _joint_loss(params, y, h, loss_fn, unpack, None if custom else n_valid)

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

    run_all takes a uniform (x0_batch, lower, upper, y, n_valid); n_valid (the
    real, pre-padding series length) enables the padding/masking scheme in
    _pad_to_bucket for built-in losses and is ignored for custom callables --
    see _compile_objective_and_grad.
    """
    objective, grad = _compile_objective_and_grad(h, loss_key, kind)
    solver = jaxopt.LBFGSB(
        fun=lambda p, y, n_valid: (objective(p, y, n_valid), grad(p, y, n_valid)),
        value_and_grad=True,
        maxiter=_LBFGSB_MAXITER,
        tol=_LBFGSB_TOL,
    )

    def run_all(x0_batch: Any, lower: Any, upper: Any, y: Any, n_valid: Any) -> tuple[Any, Any]:
        def run_one(x0: Any) -> Any:
            return solver.run(x0, (lower, upper), y, n_valid)

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


def _bucket_length(n: int) -> int:
    """Smallest power of two >= n."""
    return 1 << (n - 1).bit_length()


def _pad_to_bucket(y: np.ndarray) -> np.ndarray:
    """Right-pad y (by repeating its last value) to the next power-of-two length.

    Rolling-origin backtests refit at every expanding-window length (n, n+h,
    n+2h, ...), and each distinct length used to force a fresh XLA compile in
    the caches above (#17) since a fresh length never repeats within one
    backtest. Rounding up to a shared bucket length collapses the distinct
    shapes a backtest produces from O(n) to O(log n), so far more folds land
    on an already-compiled shape.

    This is safe because the recursion is causal: states at any real (t <
    len(y)) origin only ever depend on y[:t], so padding at the tail can't
    change them, and the padded origins themselves are excluded from the
    loss via the n_valid mask threaded through _forecast_matrix /
    _forecast_matrix_seasonal -- so fitted parameters are unaffected.
    """
    n_pad = _bucket_length(len(y))
    if n_pad == len(y):
        return y
    return np.pad(y, (0, n_pad - len(y)), mode="edge")


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


# --- Hand-derived (NumPy, no JAX) fitting path for the non-seasonal kinds ---
#
# Replaces jax.grad/jaxopt for SES/Holt/DampedHolt (#18). _loss_value_np and
# _loss_grad_np are the only pieces that differ per built-in loss (mse/mae/
# huber/pinball); they're kept recursion-agnostic so #19 (seasonal models)
# reuses them unchanged for its own d(loss)/d(forecast) step.


def _masked_mean_np(x: np.ndarray, mask: np.ndarray | None) -> float:
    """Mirrors _masked_mean, but plain NumPy."""
    if mask is None:
        return float(x.mean())
    weighted = np.where(mask[:, None], x, 0.0)
    return float(weighted.sum() / (mask.sum() * x.shape[-1]))


def _loss_value_np(
    loss_key: tuple[str, float, float],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    mask: np.ndarray | None = None,
) -> float:
    """loss(y_true, y_pred), for a resolved built-in loss_key (see _loss_key)."""
    name, delta, tau = loss_key
    if name == "mse":
        return _masked_mean_np((y_pred - y_true) ** 2, mask)
    if name == "mae":
        return _masked_mean_np(np.abs(y_pred - y_true), mask)
    if name == "huber":
        r = y_pred - y_true
        abs_r = np.abs(r)
        quadratic = np.minimum(abs_r, delta)
        linear = abs_r - quadratic
        return _masked_mean_np(0.5 * quadratic**2 + delta * linear, mask)
    if name == "pinball":
        e = y_true - y_pred
        return _masked_mean_np(np.maximum(tau * e, (tau - 1) * e), mask)
    raise ValueError(f"Unknown loss {name!r}")  # pragma: no cover -- _loss_key already validated


def _loss_grad_np(
    loss_key: tuple[str, float, float],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """d(loss)/d(y_pred), same shape as y_pred, for a resolved built-in loss_key.

    The one piece that changes between loss types; shared by the non-seasonal
    (here) and seasonal (#19) BPTT passes.
    """
    name, delta, tau = loss_key
    n = (mask.sum() if mask is not None else y_pred.shape[0]) * y_pred.shape[-1]
    scale = (mask[:, None] if mask is not None else 1.0) / n
    if name == "mse":
        g = 2.0 * (y_pred - y_true)
    elif name == "mae":
        g = np.sign(y_pred - y_true)
    elif name == "huber":
        g = np.clip(y_pred - y_true, -delta, delta)
    elif name == "pinball":
        g = np.where(y_true >= y_pred, -tau, 1.0 - tau)
    else:
        raise ValueError(f"Unknown loss {name!r}")  # pragma: no cover -- validated by _loss_key
    return g * scale


def _origin_mask_np(n_origins: int, n_valid: int, h: int) -> np.ndarray:
    """Mirrors _origin_mask, but plain NumPy (n_valid is never None here)."""
    return np.arange(n_origins) < (n_valid - h + 1)


def _forecast_multiplier_np(phi: float, h: int) -> np.ndarray:
    """Mirrors _forecast_multiplier, but plain NumPy."""
    steps = np.arange(1, h + 1)
    return np.cumsum(phi**steps)


def _states_np(
    alpha: float, beta: float, phi: float, level0: float, trend0: float, y: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Mirrors _states, but a plain Python/NumPy loop instead of jax.lax.scan."""
    t_max = len(y)
    levels = np.empty(t_max + 1)
    trends = np.empty(t_max + 1)
    levels[0], trends[0] = level0, trend0
    for t in range(t_max):
        level, trend = levels[t], trends[t]
        new_level = alpha * y[t] + (1 - alpha) * (level + phi * trend)
        new_trend = beta * (new_level - level) + (1 - beta) * phi * trend
        levels[t + 1], trends[t + 1] = new_level, new_trend
    return levels, trends


def _holt_forward_np(
    params: np.ndarray, y: np.ndarray, h: int, kind: str
) -> tuple[float, float, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(alpha, beta, phi, levels, trends, multiplier, forecasts, actuals) for one params guess."""
    alpha, beta, phi, level0, trend0 = _UNPACK_BY_KIND[kind](params)
    levels, trends = _states_np(alpha, beta, phi, level0, trend0, y)
    n_origins = len(y) - h + 1
    multiplier = _forecast_multiplier_np(phi, h)
    forecasts = levels[:n_origins, None] + multiplier[None, :] * trends[:n_origins, None]
    actuals = np.stack([y[t : t + h] for t in range(n_origins)])
    return alpha, beta, phi, levels, trends, multiplier, forecasts, actuals


def _states_bptt_grad_np(
    alpha: float,
    beta: float,
    phi: float,
    levels: np.ndarray,
    trends: np.ndarray,
    y: np.ndarray,
    d_forecast: np.ndarray,
    multiplier: np.ndarray,
    h: int,
) -> np.ndarray:
    """Backprop-through-time gradient of the joint loss wrt (alpha, beta, phi, level0, trend0).

    d_forecast is d(loss)/d(forecast) (see _loss_grad_np); levels/trends are
    _states_np's full output. Mirrors what jax.grad(_joint_loss) computed
    automatically for the jax path, including the multiplier's own
    dependence on phi (forecast = level + multiplier(phi) * trend).
    """
    n_origins = d_forecast.shape[0]
    gl_direct = d_forecast.sum(axis=1)
    gt_direct = d_forecast @ multiplier
    steps = np.arange(1, h + 1)
    d_multiplier_d_phi = np.cumsum(steps * phi ** (steps - 1))
    d_phi = float(np.sum(trends[:n_origins] * (d_forecast @ d_multiplier_d_phi)))

    d_alpha = 0.0
    d_beta = 0.0
    gl_next = 0.0
    gt_next = 0.0
    for t in range(n_origins - 1, 0, -1):
        gl_t = gl_direct[t] + gl_next
        gt_t = gt_direct[t] + gt_next
        level_prev, trend_prev = levels[t - 1], trends[t - 1]
        y_prev = y[t - 1]
        pre_activation = y_prev - level_prev - phi * trend_prev

        d_alpha += (gl_t + beta * gt_t) * pre_activation
        d_beta += gt_t * (levels[t] - level_prev - phi * trend_prev)
        d_phi += trend_prev * (gl_t * (1 - alpha) + gt_t * (1 - alpha * beta))

        gl_next = gl_t * (1 - alpha) - gt_t * alpha * beta
        gt_next = gl_t * (1 - alpha) * phi + gt_t * phi * (1 - alpha * beta)

    d_level0 = gl_direct[0] + gl_next
    d_trend0 = gt_direct[0] + gt_next
    return np.array([d_alpha, d_beta, d_phi, d_level0, d_trend0])


def _pack_grad_undamped(d: np.ndarray) -> np.ndarray:
    d_alpha, d_beta, _d_phi, d_level0, d_trend0 = d
    return np.array([d_alpha, d_beta, d_level0, d_trend0])


def _pack_grad_damped(d: np.ndarray) -> np.ndarray:
    return d


def _pack_grad_ses(d: np.ndarray) -> np.ndarray:
    d_alpha, _d_beta, _d_phi, d_level0, _d_trend0 = d
    return np.array([d_alpha, d_level0])


_PACK_GRAD_BY_KIND = {
    "undamped": _pack_grad_undamped,
    "damped": _pack_grad_damped,
    "ses": _pack_grad_ses,
}


def _holt_objective_and_grad(
    params: np.ndarray,
    y: np.ndarray,
    h: int,
    kind: str,
    loss_key: tuple[str, float, float],
    n_valid: int | None = None,
) -> tuple[float, np.ndarray]:
    """(loss, grad) for the plain (non-seasonal) Holt recursion, hand-derived (#18).

    grad has the same length/layout as params (per _UNPACK_BY_KIND[kind]).
    loss_key must be a resolved built-in key (see _loss_key) -- callers fall
    back to scipy's own finite-difference jac for custom callables, since
    those aren't differentiated here.
    """
    alpha, beta, phi, levels, trends, multiplier, forecasts, actuals = _holt_forward_np(
        params, y, h, kind
    )
    n_origins = forecasts.shape[0]
    mask = _origin_mask_np(n_origins, n_valid, h) if n_valid is not None else None
    value = _loss_value_np(loss_key, actuals, forecasts, mask)
    d_forecast = _loss_grad_np(loss_key, actuals, forecasts, mask)
    grad5 = _states_bptt_grad_np(alpha, beta, phi, levels, trends, y, d_forecast, multiplier, h)
    return value, _PACK_GRAD_BY_KIND[kind](grad5)


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
        custom = callable(loss_key)
        y_fit = y if custom else _pad_to_bucket(y)
        n_valid = len(y)

        def objective(p: np.ndarray) -> Any:
            if custom:
                # No hand-derived gradient for an arbitrary user loss; scipy's
                # jac=False below falls back to its own finite-difference
                # estimate (slower/less precise than the old jax.grad, but
                # this path no longer imports jax at all -- see #18).
                *_, forecasts, actuals = _holt_forward_np(p, y_fit, self.h, self._kind)
                return float(loss_key(actuals, forecasts))
            return _holt_objective_and_grad(p, y_fit, self.h, self._kind, loss_key, n_valid)

        # (level0, trend0) heuristic init; SES._n_free_params == 1 drops trend0.
        free_guess = _heuristic_init(y)[: self._n_free_params]
        lower = [_BOUND_EPS] * self._n_trend_params + [-np.inf] * self._n_free_params
        upper = [1 - _BOUND_EPS] * self._n_trend_params + [np.inf] * self._n_free_params
        if self._kind == "damped":
            upper[2] = _PHI_UPPER  # phi: cap at the classical damping convention
        bounds = list(zip(lower, upper, strict=True))

        best = None
        for start in _start_grid(self.n_starts, self._n_trend_params):
            x0 = np.concatenate([start, free_guess])
            result = minimize(objective, x0, method="L-BFGS-B", jac=not custom, bounds=bounds)
            if best is None or result.fun < best.fun:
                best = result

        assert best is not None  # _start_grid always yields at least one start
        self.params_ = best.x
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
            forecasts, actuals, _mask = _forecast_matrix_seasonal(
                self.params_, y_jax, self.h, unpack, self.seasonal_period, multiplicative
            )
        else:
            unpack = _UNPACK_BY_KIND[self._kind]
            forecasts, actuals, _mask = _forecast_matrix(self.params_, y_jax, self.h, unpack)
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
        custom = callable(loss_key)
        # Multiplicative seasonal states evolve via y_t/s_old, y_t/level (not a
        # convex combination like every other kind), so unlike the additive/
        # non-seasonal recursions -- provably bounded, spectral radius < 1
        # everywhere in the alpha/beta/(phi) box -- iterating it hundreds of
        # extra steps on constant padding can blow up toward +-inf/0 well
        # within float64 range at plausible box-valid multi-start corners
        # (confirmed by direct simulation). Padding is skipped here; the
        # bucketing benefit is only safe for the additive/non-seasonal kinds.
        y_fit = y if (custom or multiplicative) else _pad_to_bucket(y)
        y_jax = jnp.asarray(y_fit)
        objective, grad = _compiled_objective_and_grad(self.h, loss_key, self._kind, m, len(y_fit))
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
                lambda p: float(objective(p, y_jax, len(y))),
                x0,
                jac=lambda p: np.asarray(grad(p, y_jax, len(y))),
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
