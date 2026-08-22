# cython: boundscheck=False, wraparound=False, cdivision=True
"""Cython port of holt.py's non-seasonal (#22) and seasonal (#23) objective+gradient.

Generalizes the issue #21 spike (`spikes/issue-21-cython-vs-rust/cython_proto/
holt_hot.pyx`, undamped + mse only) into two cores:

- objective_and_grad: mirrors _holt_forward_np + _states_bptt_grad_np +
  _holt_objective_and_grad, covering all 12 (kind, loss) combinations.
- seasonal_objective_and_grad: mirrors _seasonal_states_np +
  _seasonal_states_bptt_grad_np + _seasonal_objective_and_grad, covering all
  16 (seasonal-kind, loss) combinations, including the multiplicative combine
  (y_t/s_old, y_t/new_level) which changes both forward and backward passes,
  not just the forward formula.

Shared conventions between the two cores:

- phi is always a genuine free parameter (never hardcoded to 1.0 the way the
  spike did): the BPTT already computes d_phi unconditionally. Callers
  wanting a fixed phi (undamped/ses/seasonal/seasonal_mult) just pass
  phi=1.0 in and drop the corresponding grad component afterwards --
  mirroring _UNPACK_BY_KIND/_PACK_GRAD_BY_KIND and
  _UNPACK_SEASONAL_BY_KIND/_PACK_GRAD_SEASONAL_BY_KIND -- rather than either
  core special-casing any kind.
- multiplicative is a runtime bool/int flag threaded through the seasonal
  recursion, not a compile-time specialization -- there is only ever one
  seasonal function.
- The built-in loss is picked at runtime via loss_id, mirroring
  _loss_value_np/_loss_grad_np's per-name branches:
      0 = mse, 1 = mae, 2 = huber, 3 = pinball,
      4 = poisson, 5 = negbinom, 6 = tweedie, 7 = expectile, 8 = logcosh
  delta/tau are only read by huber/pinball/expectile; nb_alpha/tw_p are only
  read by negbinom/tweedie respectively (#32). Both cores dispatch through
  the same _loss_and_grad helper below.

Custom callable losses are out of scope here (they stay on the NumPy path,
per #18's `custom` branch in Holt.fit()/HoltWinters.fit()); these cores only
serve the 9 built-ins.

Missing values (#35/#36): a NaN in y is skip-and-carry in the recursion
(mirrors _states_np/_seasonal_states_np) and zero-weighted in the joint loss
whenever it lands in an actual (mirrors _nan_actual_mask), exactly like the
NumPy path -- see holt.py for the reference semantics. No padding mask is
involved (see below); NaN is the only per-element skip mechanism here.

No padding mask (n_valid == len(y) always in the current codebase, so the
mask is always all-True -- see the ADR from #21).

objective_and_grad: params = [alpha, beta, phi, level0, trend0]; returns
(loss, grad) with grad in the same 5-element layout.

seasonal_objective_and_grad: params = [alpha, beta, phi, gamma, level0,
trend0, season0[0..m-1]] (6 + m elements); returns (loss, grad) with grad in
the same (6 + m)-element layout.

The kind-specific unpack/pack down to each model's flat param/grad layout is
plain Python -- see tests/test_holt_hot.py and tests/test_seasonal_hot.py.
"""
import numpy as np
cimport numpy as np
from libc.math cimport exp, fabs, fmax, fmin, isnan, log, log1p, pow, tanh

# Matches holt.py's _BOUND_EPS -- the floor forecasts are clamped to before
# feeding a log/power (poisson/negbinom/tweedie deviances are undefined at
# forecast <= 0).
cdef double _BOUND_EPS = 1e-4


cdef inline double _xlogy(double x, double y):
    """x * log(y), but exactly 0 when x == 0 (mirrors scipy.special.xlogy)."""
    return 0.0 if x == 0.0 else x * log(y)


cdef inline double _loss_and_grad(
    double forecast, double true, int loss_id, double delta, double tau,
    double nb_alpha, double tw_p, double *g
):
    """loss(forecast, true) and d(loss)/d(forecast), for one (t, k) pair.

    Mirrors _loss_value_np/_loss_grad_np's per-name formulas; shared by both
    cores below so the loss dispatch is written exactly once.
    """
    cdef double resid, abs_r, quad, lin, e, loss, mu, w_asym, r
    resid = forecast - true
    if loss_id == 0:  # mse
        loss = resid * resid
        g[0] = 2.0 * resid
    elif loss_id == 1:  # mae
        loss = fabs(resid)
        g[0] = 1.0 if resid > 0 else (-1.0 if resid < 0 else 0.0)
    elif loss_id == 2:  # huber
        abs_r = fabs(resid)
        quad = fmin(abs_r, delta)
        lin = abs_r - quad
        loss = 0.5 * quad * quad + delta * lin
        g[0] = fmin(fmax(resid, -delta), delta)
    elif loss_id == 3:  # pinball
        e = true - forecast
        loss = fmax(tau * e, (tau - 1.0) * e)
        g[0] = -tau if true >= forecast else (1.0 - tau)
    elif loss_id == 4:  # poisson deviance
        mu = fmax(forecast, _BOUND_EPS)
        loss = 2.0 * (_xlogy(true, true / mu) - (true - mu))
        g[0] = 2.0 * (1.0 - true / mu) if forecast > _BOUND_EPS else 0.0
    elif loss_id == 5:  # negbinom deviance
        mu = fmax(forecast, _BOUND_EPS)
        loss = 2.0 * (
            _xlogy(true, true / mu)
            - (true + 1.0 / nb_alpha) * log((1.0 + nb_alpha * true) / (1.0 + nb_alpha * mu))
        )
        g[0] = 2.0 * (mu - true) / (mu * (1.0 + nb_alpha * mu)) if forecast > _BOUND_EPS else 0.0
    elif loss_id == 6:  # tweedie deviance
        mu = fmax(forecast, _BOUND_EPS)
        loss = 2.0 * (
            pow(true, 2.0 - tw_p) / ((1.0 - tw_p) * (2.0 - tw_p))
            - true * pow(mu, 1.0 - tw_p) / (1.0 - tw_p)
            + pow(mu, 2.0 - tw_p) / (2.0 - tw_p)
        )
        g[0] = 2.0 * pow(mu, -tw_p) * (mu - true) if forecast > _BOUND_EPS else 0.0
    elif loss_id == 7:  # expectile
        w_asym = tau if true >= forecast else (1.0 - tau)
        loss = w_asym * resid * resid
        g[0] = 2.0 * w_asym * resid
    else:  # 8 = logcosh
        r = fabs(resid)
        loss = r + log1p(exp(-2.0 * r)) - log(2.0)
        g[0] = tanh(resid)
    return loss


cdef inline void _multiplier_and_dphi(
    double phi, int h, double[:] multiplier, double[:] d_multiplier_dphi
):
    """Fill multiplier[k] = sum_{i=1}^{k+1} phi**i and its d/dphi, in place.

    Mirrors _forecast_multiplier_np/_forecast_multiplier_grad_np, built
    incrementally in one pass; shared by both cores below.
    """
    cdef int k, i
    cdef double phi_pow = 1.0  # phi**(i-1), starts at phi**0
    cdef double cur_mult = 0.0
    cdef double cur_dmult = 0.0
    for k in range(h):
        i = k + 1
        cur_dmult += i * phi_pow
        phi_pow *= phi  # now phi**i
        cur_mult += phi_pow
        multiplier[k] = cur_mult
        d_multiplier_dphi[k] = cur_dmult


def objective_and_grad(
    np.ndarray[np.float64_t, ndim=1] params,
    np.ndarray[np.float64_t, ndim=1] y,
    int h,
    int loss_id,
    double delta,
    double tau,
    double nb_alpha=1.0,
    double tw_p=1.5,
    np.ndarray[np.float64_t, ndim=1] sample_weight=None,
):
    cdef bint has_weight = sample_weight is not None
    cdef double alpha = params[0]
    cdef double beta = params[1]
    cdef double phi = params[2]
    cdef double level0 = params[3]
    cdef double trend0 = params[4]
    cdef int t_max = y.shape[0]
    cdef int n_origins = t_max - h + 1
    cdef int t, k

    cdef np.ndarray[np.float64_t, ndim=1] levels = np.empty(t_max + 1)
    cdef np.ndarray[np.float64_t, ndim=1] trends = np.empty(t_max + 1)
    levels[0] = level0
    trends[0] = trend0
    cdef double level, trend, new_level, new_trend
    for t in range(t_max):
        level = levels[t]
        trend = trends[t]
        if isnan(y[t]):  # missing observation (#35): skip the update, carry state forward
            new_level = level
            new_trend = trend
        else:
            new_level = alpha * y[t] + (1 - alpha) * (level + phi * trend)
            new_trend = beta * (new_level - level) + (1 - beta) * phi * trend
        levels[t + 1] = new_level
        trends[t + 1] = new_trend

    cdef np.ndarray[np.float64_t, ndim=1] multiplier = np.empty(h)
    cdef np.ndarray[np.float64_t, ndim=1] d_multiplier_dphi = np.empty(h)
    _multiplier_and_dphi(phi, h, multiplier, d_multiplier_dphi)

    # forward + loss dispatch: forecasts[t, k] = levels[t] + multiplier[k] * trends[t],
    # actuals[t, k] = y[t+k]. Mirrors _loss_value_np/_loss_grad_np's per-name formulas.
    cdef np.ndarray[np.float64_t, ndim=2] d_forecast = np.zeros((n_origins, h))
    cdef double weight_sum = 0.0  # sum of cell weights (#39), not just a count of cells
    cdef double loss = 0.0
    cdef double forecast, true, g, w
    for t in range(n_origins):
        for k in range(h):
            true = y[t + k]
            if isnan(true):  # zero-weight this cell (#35): mirrors _nan_actual_mask
                continue
            w = sample_weight[t + k] if has_weight else 1.0
            forecast = levels[t] + multiplier[k] * trends[t]
            loss += w * _loss_and_grad(forecast, true, loss_id, delta, tau, nb_alpha, tw_p, &g)
            d_forecast[t, k] = w * g
            weight_sum += w
    loss /= weight_sum
    for t in range(n_origins):
        for k in range(h):
            d_forecast[t, k] /= weight_sum

    # BPTT, mirroring _states_bptt_grad_np with phi as a genuine free parameter
    # (including the multiplier's own dependence on phi).
    cdef np.ndarray[np.float64_t, ndim=1] gl_direct = np.empty(n_origins)
    cdef np.ndarray[np.float64_t, ndim=1] gt_direct = np.empty(n_origins)
    cdef double gl_sum, gt_sum, dphi_mult_sum
    cdef double d_phi = 0.0
    for t in range(n_origins):
        gl_sum = 0.0
        gt_sum = 0.0
        dphi_mult_sum = 0.0
        for k in range(h):
            gl_sum += d_forecast[t, k]
            gt_sum += multiplier[k] * d_forecast[t, k]
            dphi_mult_sum += d_multiplier_dphi[k] * d_forecast[t, k]
        gl_direct[t] = gl_sum
        gt_direct[t] = gt_sum
        d_phi += trends[t] * dphi_mult_sum

    cdef double d_alpha = 0.0
    cdef double d_beta = 0.0
    cdef double gl_next = 0.0
    cdef double gt_next = 0.0
    cdef double gl_t, gt_t, level_prev, trend_prev, y_prev, pre_activation
    for t in range(n_origins - 1, 0, -1):
        gl_t = gl_direct[t] + gl_next
        gt_t = gt_direct[t] + gt_next
        if isnan(y[t - 1]):  # skipped step (#35): identity backprop, no param gradient
            gl_next = gl_t
            gt_next = gt_t
            continue
        level_prev = levels[t - 1]
        trend_prev = trends[t - 1]
        y_prev = y[t - 1]
        pre_activation = y_prev - level_prev - phi * trend_prev

        d_alpha += (gl_t + beta * gt_t) * pre_activation
        d_beta += gt_t * (levels[t] - level_prev - phi * trend_prev)
        d_phi += trend_prev * (gl_t * (1 - alpha) + gt_t * (1 - alpha * beta))

        gl_next = gl_t * (1 - alpha) - gt_t * alpha * beta
        gt_next = gl_t * (1 - alpha) * phi + gt_t * phi * (1 - alpha * beta)

    cdef double d_level0 = gl_direct[0] + gl_next
    cdef double d_trend0 = gt_direct[0] + gt_next

    cdef np.ndarray[np.float64_t, ndim=1] grad = np.array(
        [d_alpha, d_beta, d_phi, d_level0, d_trend0]
    )
    return loss, grad


def seasonal_objective_and_grad(
    np.ndarray[np.float64_t, ndim=1] params,
    np.ndarray[np.float64_t, ndim=1] y,
    int h,
    int m,
    int multiplicative,
    int loss_id,
    double delta,
    double tau,
    double nb_alpha=1.0,
    double tw_p=1.5,
    np.ndarray[np.float64_t, ndim=1] sample_weight=None,
):
    """Mirrors _seasonal_states_np + _seasonal_states_bptt_grad_np +
    _seasonal_objective_and_grad.

    params = [alpha, beta, phi, gamma, level0, trend0, season0[0..m-1]];
    returns (loss, grad) with grad in the same (6 + m)-element layout.
    multiplicative is a runtime flag (0/1), not a compile-time specialization.
    """
    cdef bint has_weight = sample_weight is not None
    cdef double alpha = params[0]
    cdef double beta = params[1]
    cdef double phi = params[2]
    cdef double gamma = params[3]
    cdef double level0 = params[4]
    cdef double trend0 = params[5]
    cdef np.ndarray[np.float64_t, ndim=1] season0 = params[6 : 6 + m]
    cdef int t_max = y.shape[0]
    cdef int n_origins = t_max - h + 1
    cdef int t, k, j

    cdef np.ndarray[np.float64_t, ndim=1] levels = np.empty(t_max + 1)
    cdef np.ndarray[np.float64_t, ndim=1] trends = np.empty(t_max + 1)
    cdef np.ndarray[np.float64_t, ndim=2] season_bufs = np.empty((t_max + 1, m))
    levels[0] = level0
    trends[0] = trend0
    for j in range(m):
        season_bufs[0, j] = season0[j]

    cdef double level, trend, s_old, new_level, new_trend, new_season
    for t in range(t_max):
        level = levels[t]
        trend = trends[t]
        s_old = season_bufs[t, 0]
        if isnan(y[t]):  # missing observation (#36): skip the update, carry state forward
            new_level = level
            new_trend = trend
            new_season = s_old
        elif multiplicative:
            new_level = alpha * (y[t] / s_old) + (1 - alpha) * (level + phi * trend)
            new_trend = beta * (new_level - level) + (1 - beta) * phi * trend
            new_season = gamma * (y[t] / new_level) + (1 - gamma) * s_old
        else:
            new_level = alpha * (y[t] - s_old) + (1 - alpha) * (level + phi * trend)
            new_trend = beta * (new_level - level) + (1 - beta) * phi * trend
            new_season = gamma * (y[t] - new_level) + (1 - gamma) * s_old
        levels[t + 1] = new_level
        trends[t + 1] = new_trend
        for j in range(m - 1):
            season_bufs[t + 1, j] = season_bufs[t, j + 1]
        season_bufs[t + 1, m - 1] = new_season

    cdef np.ndarray[np.float64_t, ndim=1] multiplier = np.empty(h)
    cdef np.ndarray[np.float64_t, ndim=1] d_multiplier_dphi = np.empty(h)
    _multiplier_and_dphi(phi, h, multiplier, d_multiplier_dphi)

    # forward + loss dispatch: trend_part[t,k] = levels[t] + multiplier[k]*trends[t],
    # seasonal_part[t,k] = season_bufs[t, k % m], forecast = trend_part (+|*) seasonal_part.
    # Mirrors _seasonal_states_bptt_grad_np's dF_dlevel/dF_dtrend/dF_dbuf_weight,
    # accumulated directly into gl_direct/gt_direct/gb_direct instead of stored
    # as full (n_origins, h) matrices.
    cdef np.ndarray[np.float64_t, ndim=1] gl_direct = np.zeros(n_origins)
    cdef np.ndarray[np.float64_t, ndim=1] gt_direct = np.zeros(n_origins)
    cdef np.ndarray[np.float64_t, ndim=2] gb_direct = np.zeros((n_origins, m))
    cdef double weight_sum = 0.0  # sum of cell weights (#39), not just a count of cells
    cdef double loss = 0.0
    cdef double forecast, true, g, w, trend_part_val, seasonal_part_val, mult_sum
    cdef double d_phi = 0.0
    for t in range(n_origins):
        mult_sum = 0.0
        for k in range(h):
            true = y[t + k]
            if isnan(true):  # zero-weight this cell (#36): mirrors _nan_actual_mask
                continue
            w = sample_weight[t + k] if has_weight else 1.0
            trend_part_val = levels[t] + multiplier[k] * trends[t]
            j = k % m
            seasonal_part_val = season_bufs[t, j]
            if multiplicative:
                forecast = trend_part_val * seasonal_part_val
            else:
                forecast = trend_part_val + seasonal_part_val
            loss += w * _loss_and_grad(forecast, true, loss_id, delta, tau, nb_alpha, tw_p, &g)
            g *= w
            weight_sum += w

            if multiplicative:
                gl_direct[t] += g * seasonal_part_val
                gt_direct[t] += g * multiplier[k] * seasonal_part_val
                gb_direct[t, j] += g * trend_part_val
                mult_sum += g * seasonal_part_val * d_multiplier_dphi[k]
            else:
                gl_direct[t] += g
                gt_direct[t] += g * multiplier[k]
                gb_direct[t, j] += g
                mult_sum += g * d_multiplier_dphi[k]
        d_phi += trends[t] * mult_sum

    loss /= weight_sum
    for t in range(n_origins):
        gl_direct[t] /= weight_sum
        gt_direct[t] /= weight_sum
        for j in range(m):
            gb_direct[t, j] /= weight_sum
    d_phi /= weight_sum

    # BPTT, mirroring _seasonal_states_bptt_grad_np: season0 is a free m-vector
    # too, and new_level feeds both new_trend and new_season each step, so the
    # incoming gradient on new_level (gl_full) must fold in both consumers
    # before deriving parameter grads and backflowing into the previous step.
    cdef double d_alpha = 0.0
    cdef double d_beta = 0.0
    cdef double d_gamma = 0.0
    cdef double gl_next = 0.0
    cdef double gt_next = 0.0
    cdef np.ndarray[np.float64_t, ndim=1] gb_next = np.zeros(m)
    cdef np.ndarray[np.float64_t, ndim=1] gb_cur = np.empty(m)
    cdef np.ndarray[np.float64_t, ndim=1] next_gb = np.empty(m)
    cdef double gl, gt, gS, level_prev, trend_prev, s_old2, y_prev, new_level2
    cdef double f, g_val, df_ds_old, dg_dlevel, gl_full
    for t in range(n_origins - 1, 0, -1):
        gl = gl_direct[t] + gl_next
        gt = gt_direct[t] + gt_next
        for j in range(m):
            gb_cur[j] = gb_direct[t, j] + gb_next[j]

        if isnan(y[t - 1]):
            # Skipped step (#36): level/trend pass through unchanged, and the
            # season buffer just rotates (s_old is re-appended as new_season
            # verbatim) -- no param gradient, and s_old's only outgoing edge
            # is that rotation, so its gradient is the incoming tail gradient.
            gl_next = gl
            gt_next = gt
            next_gb[0] = gb_cur[m - 1]
            for j in range(1, m):
                next_gb[j] = gb_cur[j - 1]
            for j in range(m):
                gb_next[j] = next_gb[j]
            continue

        level_prev = levels[t - 1]
        trend_prev = trends[t - 1]
        s_old2 = season_bufs[t - 1, 0]
        y_prev = y[t - 1]
        new_level2 = levels[t]

        if multiplicative:
            f = y_prev / s_old2
            g_val = y_prev / new_level2
            df_ds_old = -y_prev / (s_old2 * s_old2)
            dg_dlevel = -y_prev / (new_level2 * new_level2)
        else:
            f = y_prev - s_old2
            g_val = y_prev - new_level2
            df_ds_old = -1.0
            dg_dlevel = -1.0

        gS = gb_cur[m - 1]  # gradient wrt the season value new_season_{t-1} just appended
        gl_full = gl + beta * gt + gS * gamma * dg_dlevel

        d_alpha += gl_full * (f - (level_prev + phi * trend_prev))
        d_beta += gt * (new_level2 - level_prev - phi * trend_prev)
        d_phi += trend_prev * (gl_full * (1 - alpha) + gt * (1 - beta))
        d_gamma += gS * (g_val - s_old2)

        gl_next = gl_full * (1 - alpha) - beta * gt
        gt_next = phi * (gl_full * (1 - alpha) + gt * (1 - beta))
        next_gb[0] = gl_full * alpha * df_ds_old + gS * (1 - gamma)
        for j in range(1, m):
            next_gb[j] = gb_cur[j - 1]
        for j in range(m):
            gb_next[j] = next_gb[j]

    cdef double d_level0 = gl_direct[0] + gl_next
    cdef np.ndarray[np.float64_t, ndim=1] d_season0 = np.empty(m)
    for j in range(m):
        d_season0[j] = gb_direct[0, j] + gb_next[j]
    cdef double d_trend0 = gt_direct[0] + gt_next

    cdef np.ndarray[np.float64_t, ndim=1] grad = np.empty(6 + m)
    grad[0] = d_alpha
    grad[1] = d_beta
    grad[2] = d_phi
    grad[3] = d_gamma
    grad[4] = d_level0
    grad[5] = d_trend0
    for j in range(m):
        grad[6 + j] = d_season0[j]
    return loss, grad
