"""Holt's linear trend model, fit by minimizing a joint h-step-ahead loss."""

import itertools
from collections.abc import Callable
from typing import Any, Self

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm

try:
    from multi_ets import _holt_hot  # ty: ignore[unresolved-import]
except ImportError:  # extension not built -- optional `[fast]` extra, see docs/adr/0002
    _holt_hot = None
_HOLT_HOT_AVAILABLE = _holt_hot is not None

_BOUND_EPS = 1e-4
# Standard classical-exponential-smoothing damping-parameter range (Hyndman &
# Athanasopoulos); phi
# approaching 1 means "no damping", which defeats the point of a damped model.
_PHI_UPPER = 0.98


def _loss_key(loss: str | Callable[..., Any], delta: float, tau: float) -> Any:
    """Hashable identity for a loss choice; a resolved built-in is (name, delta, tau)."""
    if callable(loss):
        return loss
    if loss not in ("mse", "mae", "huber", "pinball"):
        raise ValueError(
            f"Unknown loss {loss!r}; supported: mse, mae, huber, pinball, or a callable"
        )
    if loss == "huber" and delta <= 0:
        raise ValueError(f"huber delta must be positive, got {delta}")
    if loss == "pinball" and not 0 < tau < 1:
        raise ValueError(f"pinball tau must be in (0, 1), got {tau}")
    return (loss, delta, tau)


# loss_id convention shared with _holt_hot.pyx (0=mse, 1=mae, 2=huber, 3=pinball).
_LOSS_ID = {"mse": 0, "mae": 1, "huber": 2, "pinball": 3}


def _resolve_sample_weight(sample_weight: Any, y: np.ndarray, custom: bool) -> np.ndarray | None:
    """Validate fit()'s sample_weight (#25) against y; None if not given."""
    if sample_weight is None:
        return None
    if custom:
        raise ValueError("sample_weight is not supported with a custom callable loss")
    weight = np.asarray(sample_weight, dtype=float)
    if weight.shape != y.shape:
        raise ValueError(
            f"sample_weight must have the same length as y ({len(y)}), got {len(weight)}"
        )
    if np.any(weight < 0):
        raise ValueError("sample_weight must be non-negative")
    if weight.sum() <= 0:
        raise ValueError("sample_weight must contain at least one positive weight")
    return weight


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


def _weight_matrix_np(sample_weight: np.ndarray, n_origins: int, h: int) -> np.ndarray:
    """(n_origins, h) weight per (origin, horizon-step) cell: weight[t, k] = sample_weight[t + k].

    Mirrors actuals's construction (np.stack([y[t : t + h] ...])) so that the
    weight on a forecast-vs-actual term always matches the weight on the
    observation y[t + k] it actually scores (#25) -- a given observation gets
    the same weight in every origin window it appears in, not just the one
    where it's the origin.
    """
    return np.stack([sample_weight[t : t + h] for t in range(n_origins)])


def _full_weight(
    shape: tuple[int, int], mask: np.ndarray | None, weight: np.ndarray | None
) -> np.ndarray | None:
    """Combine the origin mask and sample_weight (#25) into one (n_origins, h) weight array."""
    if mask is None and weight is None:
        return None
    w = np.ones(shape)
    if mask is not None:
        w = w * mask[:, None]
    if weight is not None:
        w = w * weight
    return w


def _weighted_zero_safe(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Elementwise x * w, but exactly 0 wherever w == 0 (0 * inf/nan would otherwise be nan)."""
    with np.errstate(invalid="ignore"):
        return np.where(w > 0, x * w, 0.0)


def _masked_mean_np(
    x: np.ndarray, mask: np.ndarray | None, weight: np.ndarray | None = None
) -> float:
    """Mirrors _masked_mean, but plain NumPy; weight reweights each observation's contribution."""
    w = _full_weight(x.shape, mask, weight)
    if w is None:
        return float(x.mean())
    return float(_weighted_zero_safe(x, w).sum() / w.sum())


def _loss_value_np(
    loss_key: tuple[str, float, float],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    mask: np.ndarray | None = None,
    weight: np.ndarray | None = None,
) -> float:
    """loss(y_true, y_pred), for a resolved built-in loss_key (see _loss_key).

    weight (#25), shaped like y_true/y_pred, reweights each observation's
    contribution, e.g. for recency weighting; None means every term counts
    equally. See _weight_matrix_np for how a per-y-element sample_weight
    becomes this per-(origin, horizon-step) shape.
    """
    name, delta, tau = loss_key
    if name == "mse":
        return _masked_mean_np((y_pred - y_true) ** 2, mask, weight)
    if name == "mae":
        return _masked_mean_np(np.abs(y_pred - y_true), mask, weight)
    if name == "huber":
        r = y_pred - y_true
        abs_r = np.abs(r)
        quadratic = np.minimum(abs_r, delta)
        linear = abs_r - quadratic
        return _masked_mean_np(0.5 * quadratic**2 + delta * linear, mask, weight)
    if name == "pinball":
        e = y_true - y_pred
        return _masked_mean_np(np.maximum(tau * e, (tau - 1) * e), mask, weight)
    raise ValueError(f"Unknown loss {name!r}")  # pragma: no cover -- _loss_key already validated


def _loss_grad_np(
    loss_key: tuple[str, float, float],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    mask: np.ndarray | None = None,
    weight: np.ndarray | None = None,
) -> np.ndarray:
    """d(loss)/d(y_pred), same shape as y_pred, for a resolved built-in loss_key.

    The one piece that changes between loss types; shared by the non-seasonal
    (here) and seasonal (#19) BPTT passes. weight: see _loss_value_np.
    """
    name, delta, tau = loss_key
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
    w = _full_weight(y_pred.shape, mask, weight)
    if w is None:
        return g / (y_pred.shape[0] * y_pred.shape[-1])
    return _weighted_zero_safe(g, w) / w.sum()


def _origin_mask_np(n_origins: int, n_valid: int, h: int) -> np.ndarray:
    """Mirrors _origin_mask, but plain NumPy (n_valid is never None here)."""
    return np.arange(n_origins) < (n_valid - h + 1)


def _forecast_multiplier_np(phi: float, h: int) -> np.ndarray:
    """Mirrors _forecast_multiplier, but plain NumPy."""
    steps = np.arange(1, h + 1)
    return np.cumsum(phi**steps)


def _forecast_multiplier_grad_np(phi: float, h: int) -> np.ndarray:
    """d(_forecast_multiplier_np(phi, h))/d(phi); shared by both BPTT passes below."""
    steps = np.arange(1, h + 1)
    return np.cumsum(steps * phi ** (steps - 1))


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
    d_multiplier_d_phi = _forecast_multiplier_grad_np(phi, h)
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
    sample_weight: np.ndarray | None = None,
) -> tuple[float, np.ndarray]:
    """(loss, grad) for the plain (non-seasonal) Holt recursion, hand-derived (#18).

    grad has the same length/layout as params (per _UNPACK_BY_KIND[kind]).
    loss_key must be a resolved built-in key (see _loss_key) -- callers fall
    back to scipy's own finite-difference jac for custom callables, since
    those aren't differentiated here. sample_weight (#25), if given, is one
    weight per element of y; each loss term is weighted by the sample_weight
    of the specific y observation it scores (see _weight_matrix_np), so an
    observation is weighted consistently across every origin window it falls
    into, not just the one where it's the origin.
    """
    alpha, beta, phi, levels, trends, multiplier, forecasts, actuals = _holt_forward_np(
        params, y, h, kind
    )
    n_origins = forecasts.shape[0]
    mask = _origin_mask_np(n_origins, n_valid, h) if n_valid is not None else None
    weight = _weight_matrix_np(sample_weight, n_origins, h) if sample_weight is not None else None
    value = _loss_value_np(loss_key, actuals, forecasts, mask, weight)
    d_forecast = _loss_grad_np(loss_key, actuals, forecasts, mask, weight)
    grad5 = _states_bptt_grad_np(alpha, beta, phi, levels, trends, y, d_forecast, multiplier, h)
    return value, _PACK_GRAD_BY_KIND[kind](grad5)


# --- Hand-derived (NumPy, no JAX) fitting path for the seasonal kinds (#19) ---
#
# Reuses _loss_value_np/_loss_grad_np (#18) unchanged for the outer
# d(loss)/d(forecast) step; only the recursion this backprops through differs
# (_seasonal_states_np instead of _states_np), including the multiplicative
# combine, which changes the local partial derivatives at every step
# (division rules, not the additive case's linear combination) -- not just
# the forward formula.


def _seasonal_states_np(
    alpha: float,
    beta: float,
    phi: float,
    gamma: float,
    level0: float,
    trend0: float,
    season0: np.ndarray,
    y: np.ndarray,
    m: int,
    multiplicative: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mirrors _seasonal_states, but a plain Python/NumPy loop instead of jax.lax.scan."""
    t_max = len(y)
    levels = np.empty(t_max + 1)
    trends = np.empty(t_max + 1)
    season_bufs = np.empty((t_max + 1, m))
    levels[0], trends[0] = level0, trend0
    season_bufs[0] = season0
    for t in range(t_max):
        level, trend = levels[t], trends[t]
        s_old = season_bufs[t, 0]
        if multiplicative:
            new_level = alpha * (y[t] / s_old) + (1 - alpha) * (level + phi * trend)
            new_trend = beta * (new_level - level) + (1 - beta) * phi * trend
            new_season = gamma * (y[t] / new_level) + (1 - gamma) * s_old
        else:
            new_level = alpha * (y[t] - s_old) + (1 - alpha) * (level + phi * trend)
            new_trend = beta * (new_level - level) + (1 - beta) * phi * trend
            new_season = gamma * (y[t] - new_level) + (1 - gamma) * s_old
        levels[t + 1] = new_level
        trends[t + 1] = new_trend
        season_bufs[t + 1, :-1] = season_bufs[t, 1:]
        season_bufs[t + 1, -1] = new_season
    return levels, trends, season_bufs


def _seasonal_forward_np(
    params: np.ndarray, y: np.ndarray, h: int, kind: str, m: int
) -> tuple[Any, ...]:
    """(alpha, beta, phi, gamma, levels, trends, season_bufs, multiplier, trend_part, ...).

    seasonal_part, forecasts, actuals -- for one params guess. Mirrors _holt_forward_np.
    """
    alpha, beta, phi, gamma, level0, trend0, season0 = _UNPACK_SEASONAL_BY_KIND[kind](params, m)
    multiplicative = kind in _MULTIPLICATIVE_KINDS
    levels, trends, season_bufs = _seasonal_states_np(
        alpha, beta, phi, gamma, level0, trend0, season0, y, m, multiplicative
    )
    n_origins = len(y) - h + 1
    multiplier = _forecast_multiplier_np(phi, h)
    trend_part = levels[:n_origins, None] + multiplier[None, :] * trends[:n_origins, None]
    seasonal_idx = np.arange(h) % m
    seasonal_part = season_bufs[:n_origins][:, seasonal_idx]
    forecasts = trend_part * seasonal_part if multiplicative else trend_part + seasonal_part
    actuals = np.stack([y[t : t + h] for t in range(n_origins)])
    return (
        alpha,
        beta,
        phi,
        gamma,
        levels,
        trends,
        season_bufs,
        multiplier,
        trend_part,
        seasonal_part,
        forecasts,
        actuals,
    )


def _seasonal_states_bptt_grad_np(
    alpha: float,
    beta: float,
    phi: float,
    gamma: float,
    levels: np.ndarray,
    trends: np.ndarray,
    season_bufs: np.ndarray,
    y: np.ndarray,
    d_forecast: np.ndarray,
    multiplier: np.ndarray,
    trend_part: np.ndarray,
    seasonal_part: np.ndarray,
    h: int,
    m: int,
    multiplicative: bool,
) -> tuple[float, float, float, float, float, float, np.ndarray]:
    """BPTT gradient of the joint loss wrt (alpha, beta, phi, gamma, level0, ...).

    trend0, season0 -- through the seasonal recursion. d_forecast is
    d(loss)/d(forecast) (see _loss_grad_np); levels/trends/
    season_bufs are _seasonal_states_np's full output. Mirrors what
    jax.grad(_joint_loss_seasonal) computed automatically, extended for the
    season buffer (season0 is a free parameter too, so it needs its own
    gradient) and for the multiplicative combine (y_t/s_old, y_t/new_level),
    which swaps the additive case's linear local derivatives for division
    rules at every step.

    Each step's new_level feeds both new_trend and the newly appended season
    value (new_season); both of those are "outputs" of the same step, so the
    combined incoming gradient on new_level (gl_full below) must account for
    both paths before it's used to derive parameter gradients and backflow
    into the previous step's state -- mirroring how _states_bptt_grad_np
    folds new_trend's dependency on new_level into its own gl_t/gt_t combo,
    just with one more consumer.
    """
    n_origins = d_forecast.shape[0]
    idx = np.arange(h) % m

    if multiplicative:
        dF_dlevel = seasonal_part
        dF_dtrend = multiplier[None, :] * seasonal_part
        dF_dbuf_weight = trend_part
    else:
        dF_dlevel = np.ones_like(seasonal_part)
        dF_dtrend = np.broadcast_to(multiplier[None, :], seasonal_part.shape)
        dF_dbuf_weight = np.ones_like(seasonal_part)

    gl_direct = (d_forecast * dF_dlevel).sum(axis=1)
    gt_direct = (d_forecast * dF_dtrend).sum(axis=1)
    gb_direct = np.zeros((n_origins, m))
    weighted = d_forecast * dF_dbuf_weight
    for j in range(m):
        gb_direct[:, j] = weighted[:, idx == j].sum(axis=1)

    d_multiplier_d_phi = _forecast_multiplier_grad_np(phi, h)
    mult_weight = (d_forecast * (seasonal_part if multiplicative else 1.0)) @ d_multiplier_d_phi
    d_phi = float(np.sum(trends[:n_origins] * mult_weight))

    d_alpha = 0.0
    d_beta = 0.0
    d_gamma = 0.0
    gl_next = 0.0
    gt_next = 0.0
    gb_next = np.zeros(m)
    for t in range(n_origins - 1, 0, -1):
        gl = gl_direct[t] + gl_next
        gt = gt_direct[t] + gt_next
        gb = gb_direct[t] + gb_next

        level_prev, trend_prev = levels[t - 1], trends[t - 1]
        s_old = season_bufs[t - 1, 0]
        y_prev = y[t - 1]
        new_level = levels[t]

        if multiplicative:
            f = y_prev / s_old
            g = y_prev / new_level
            df_ds_old = -y_prev / s_old**2
            dg_dlevel = -y_prev / new_level**2
        else:
            f = y_prev - s_old
            g = y_prev - new_level
            df_ds_old = -1.0
            dg_dlevel = -1.0

        gS = gb[-1]  # gradient wrt the season value new_season_{t-1} just appended
        # new_level feeds both new_trend and new_season, so both paths count here.
        gl_full = gl + beta * gt + gS * gamma * dg_dlevel

        d_alpha += gl_full * (f - (level_prev + phi * trend_prev))
        d_beta += gt * (new_level - level_prev - phi * trend_prev)
        d_phi += trend_prev * (gl_full * (1 - alpha) + gt * (1 - beta))
        d_gamma += gS * (g - s_old)

        gl_next = gl_full * (1 - alpha) - beta * gt
        gt_next = phi * (gl_full * (1 - alpha) + gt * (1 - beta))
        next_gb = np.empty(m)
        next_gb[0] = gl_full * alpha * df_ds_old + gS * (1 - gamma)
        next_gb[1:] = gb[:-1]
        gb_next = next_gb

    d_level0 = gl_direct[0] + gl_next
    d_trend0 = gt_direct[0] + gt_next
    d_season0 = gb_direct[0] + gb_next
    return d_alpha, d_beta, d_phi, d_gamma, d_level0, d_trend0, d_season0


def _pack_grad_seasonal(
    d: tuple[float, float, float, float, float, float, np.ndarray],
) -> np.ndarray:
    d_alpha, d_beta, _d_phi, d_gamma, d_level0, d_trend0, d_season0 = d
    return np.concatenate([[d_alpha, d_beta, d_gamma, d_level0, d_trend0], d_season0])


def _pack_grad_damped_seasonal(
    d: tuple[float, float, float, float, float, float, np.ndarray],
) -> np.ndarray:
    d_alpha, d_beta, d_phi, d_gamma, d_level0, d_trend0, d_season0 = d
    return np.concatenate([[d_alpha, d_beta, d_phi, d_gamma, d_level0, d_trend0], d_season0])


# "_mult" kinds share the additive kind's flat-params layout and gradient
# packing (see _UNPACK_SEASONAL_BY_KIND); only the recursion differs.
_PACK_GRAD_SEASONAL_BY_KIND = {
    "seasonal": _pack_grad_seasonal,
    "damped_seasonal": _pack_grad_damped_seasonal,
    "seasonal_mult": _pack_grad_seasonal,
    "damped_seasonal_mult": _pack_grad_damped_seasonal,
}


def _seasonal_objective_and_grad(
    params: np.ndarray,
    y: np.ndarray,
    h: int,
    kind: str,
    m: int,
    loss_key: tuple[str, float, float],
    n_valid: int | None = None,
    sample_weight: np.ndarray | None = None,
) -> tuple[float, np.ndarray]:
    """(loss, grad) for the seasonal Holt-Winters recursion, hand-derived (#19).

    grad has the same length/layout as params (per _UNPACK_SEASONAL_BY_KIND[kind]).
    Mirrors _holt_objective_and_grad's non-seasonal shape; loss_key must be a
    resolved built-in key (see _loss_key) -- callers fall back to scipy's own
    finite-difference jac for custom callables, since those aren't
    differentiated here. sample_weight: see _holt_objective_and_grad.
    """
    multiplicative = kind in _MULTIPLICATIVE_KINDS
    (
        alpha,
        beta,
        phi,
        gamma,
        levels,
        trends,
        season_bufs,
        multiplier,
        trend_part,
        seasonal_part,
        forecasts,
        actuals,
    ) = _seasonal_forward_np(params, y, h, kind, m)
    n_origins = forecasts.shape[0]
    mask = _origin_mask_np(n_origins, n_valid, h) if n_valid is not None else None
    weight = _weight_matrix_np(sample_weight, n_origins, h) if sample_weight is not None else None
    value = _loss_value_np(loss_key, actuals, forecasts, mask, weight)
    d_forecast = _loss_grad_np(loss_key, actuals, forecasts, mask, weight)
    grad_tuple = _seasonal_states_bptt_grad_np(
        alpha,
        beta,
        phi,
        gamma,
        levels,
        trends,
        season_bufs,
        y,
        d_forecast,
        multiplier,
        trend_part,
        seasonal_part,
        h,
        m,
        multiplicative,
    )
    return value, _PACK_GRAD_SEASONAL_BY_KIND[kind](grad_tuple)


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

    def fit(self, y: Any, sample_weight: Any | None = None) -> Self:
        """Fit alpha/beta/level0/trend0 by joint multi-start optimization; returns self.

        sample_weight (#25), if given, is an array the same length as y that
        reweights each observation's contribution to the loss (e.g. for
        recency weighting); omitting it (or passing all-ones) reproduces the
        unweighted fit.
        """
        y = np.asarray(y, dtype=float)
        if len(y) < self.h + 1:
            raise ValueError(
                f"Series too short: need at least h + 1 = {self.h + 1} points, got {len(y)}"
            )

        loss_key = _loss_key(self.loss, self.delta, self.tau)
        custom = callable(loss_key)
        n_valid = len(y)
        weight = _resolve_sample_weight(sample_weight, y, custom)

        def objective(p: np.ndarray) -> Any:
            if custom:
                # No hand-derived gradient for an arbitrary user loss; scipy's
                # jac=False below falls back to its own finite-difference
                # estimate. This whole fit path is plain NumPy -- no jax.
                *_, forecasts, actuals = _holt_forward_np(p, y, self.h, self._kind)
                return float(loss_key(actuals, forecasts))
            if _holt_hot is not None and weight is None:
                name, delta, tau = loss_key
                full5 = np.asarray(_UNPACK_BY_KIND[self._kind](p), dtype=float)
                value, grad5 = _holt_hot.objective_and_grad(
                    full5, y, self.h, _LOSS_ID[name], delta, tau
                )
                return value, _PACK_GRAD_BY_KIND[self._kind](grad5)
            # ponytail: the compiled _holt_hot core doesn't take sample_weight
            # yet, so weighted fits always use the NumPy path; teach it to
            # _holt_hot.objective_and_grad if weighted fits need the speedup.
            return _holt_objective_and_grad(p, y, self.h, self._kind, loss_key, n_valid, weight)

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
        levels, trends = _states_np(alpha, beta, 1.0, level0, trend0, self._y)
        level, trend = float(levels[-1]), float(trends[-1])
        return level + np.arange(1, self.h + 1) * trend

    def _resid_std(self) -> np.ndarray:
        """Per-horizon-step std of (actual - forecast) across every valid training origin.

        Recomputes the forecast matrix from the fitted params, dispatching
        seasonal vs. non-seasonal by _kind exactly like the fit objective
        functions do. Not cached: it's one cheap forward pass, and caching
        risks staleness across refits.

        With very short series (few valid origins) this degenerates, e.g. a
        single origin gives 0 std and an all-zero-width interval -- accepted
        as a known limitation rather than guarded against here.
        """
        assert self.params_ is not None  # called only from predict_interval(), already checked
        if self._kind in _SEASONAL_KINDS:
            assert self.seasonal_period is not None  # set by HoltWinters for seasonal kinds
            *_, forecasts, actuals = _seasonal_forward_np(
                self.params_, self._y, self.h, self._kind, self.seasonal_period
            )
        else:
            *_, forecasts, actuals = _holt_forward_np(self.params_, self._y, self.h, self._kind)
        return np.std(actuals - forecasts, axis=0)

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
        levels, _ = _states_np(alpha, 0.0, 0.0, level0, 0.0, self._y)
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
        levels, trends = _states_np(alpha, beta, phi, level0, trend0, self._y)
        level, trend = float(levels[-1]), float(trends[-1])
        multiplier = _forecast_multiplier_np(phi, self.h)
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

    def fit(self, y: Any, sample_weight: Any | None = None) -> Self:
        """Fit trend and seasonal params by joint multi-start optimization; returns self.

        sample_weight: see Holt.fit.
        """
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
        n_valid = len(y)
        weight = _resolve_sample_weight(sample_weight, y, custom)

        def objective(p: np.ndarray) -> Any:
            if custom:
                # No hand-derived gradient for an arbitrary user loss; scipy's
                # jac=False below falls back to its own finite-difference
                # estimate. This whole fit path is plain NumPy -- no jax.
                forecasts, actuals = _seasonal_forward_np(p, y, self.h, self._kind, m)[-2:]
                return float(loss_key(actuals, forecasts))
            if _holt_hot is not None and weight is None:
                name, delta, tau = loss_key
                alpha, beta, phi, gamma, level0, trend0, season0 = _UNPACK_SEASONAL_BY_KIND[
                    self._kind
                ](p, m)
                full = np.concatenate([[alpha, beta, phi, gamma, level0, trend0], season0])
                value, grad_full = _holt_hot.seasonal_objective_and_grad(
                    full, y, self.h, m, int(multiplicative), _LOSS_ID[name], delta, tau
                )
                grad_tuple = (
                    grad_full[0],
                    grad_full[1],
                    grad_full[2],
                    grad_full[3],
                    grad_full[4],
                    grad_full[5],
                    grad_full[6:],
                )
                return value, _PACK_GRAD_SEASONAL_BY_KIND[self._kind](grad_tuple)
            # ponytail: see the non-seasonal fit() -- weighted fits skip the
            # compiled core and always use the NumPy path.
            return _seasonal_objective_and_grad(
                p, y, self.h, self._kind, m, loss_key, n_valid, weight
            )

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
            result = minimize(objective, x0, method="L-BFGS-B", jac=not custom, bounds=bounds)
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
        levels, trends, season_bufs = _seasonal_states_np(
            alpha, beta, phi, gamma, level0, trend0, season0, self._y, m, multiplicative
        )
        level, trend = float(levels[-1]), float(trends[-1])
        multiplier = _forecast_multiplier_np(phi, self.h)
        seasonal = np.asarray(season_bufs[-1])[np.arange(self.h) % m]
        trend_part = level + multiplier * trend
        return trend_part * seasonal if multiplicative else trend_part + seasonal


class DampedHoltWinters(HoltWinters):
    """Holt-Winters with additive or multiplicative seasonality and a damped trend."""

    _kind = "damped_seasonal"
    _n_trend_params = 4  # alpha, beta, phi, gamma
