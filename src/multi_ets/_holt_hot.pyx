# cython: boundscheck=False, wraparound=False, cdivision=True
"""Cython port of holt.py's non-seasonal Holt objective+gradient (issue #22).

Generalizes the issue #21 spike (`spikes/issue-21-cython-vs-rust/cython_proto/
holt_hot.pyx`, undamped + mse only) into ONE core that mirrors
_holt_forward_np + _states_bptt_grad_np + _holt_objective_and_grad from
src/multi_ets/holt.py, covering all 12 (kind, loss) combinations:

- phi is always a genuine free parameter (never hardcoded to 1.0 the way the
  spike did): _states_bptt_grad_np already computes d_phi unconditionally,
  so this core does too. Callers wanting kind="undamped"/"ses" just pass
  phi=1.0/0.0 in and drop the corresponding grad component afterwards --
  mirroring _UNPACK_BY_KIND/_PACK_GRAD_BY_KIND -- rather than this core
  special-casing any kind.
- The built-in loss is picked at runtime via loss_id, mirroring
  _loss_value_np/_loss_grad_np's per-name branches:
      0 = mse, 1 = mae, 2 = huber, 3 = pinball
  delta/tau are only read by huber/pinball respectively, same as those two
  NumPy functions.

Custom callable losses are out of scope here (they stay on the NumPy path,
per #18's `custom` branch in Holt.fit()); this core only serves the 4
built-ins.

No padding mask (n_valid == len(y) always in the current codebase, so the
mask is always all-True -- see the ADR from #21).

params = [alpha, beta, phi, level0, trend0]; returns (loss, grad) with grad
in the same 5-element layout. The kind-specific unpack/pack down to each
model's flat param/grad layout is plain Python -- see tests/test_holt_hot.py.
"""
import numpy as np
cimport numpy as np
from libc.math cimport fabs, fmax, fmin

def objective_and_grad(
    np.ndarray[np.float64_t, ndim=1] params,
    np.ndarray[np.float64_t, ndim=1] y,
    int h,
    int loss_id,
    double delta,
    double tau,
):
    cdef double alpha = params[0]
    cdef double beta = params[1]
    cdef double phi = params[2]
    cdef double level0 = params[3]
    cdef double trend0 = params[4]
    cdef int t_max = y.shape[0]
    cdef int n_origins = t_max - h + 1
    cdef int t, k, i

    cdef np.ndarray[np.float64_t, ndim=1] levels = np.empty(t_max + 1)
    cdef np.ndarray[np.float64_t, ndim=1] trends = np.empty(t_max + 1)
    levels[0] = level0
    trends[0] = trend0
    cdef double level, trend, new_level, new_trend
    for t in range(t_max):
        level = levels[t]
        trend = trends[t]
        new_level = alpha * y[t] + (1 - alpha) * (level + phi * trend)
        new_trend = beta * (new_level - level) + (1 - beta) * phi * trend
        levels[t + 1] = new_level
        trends[t + 1] = new_trend

    # multiplier[k] = sum_{i=1}^{k+1} phi**i (mirrors _forecast_multiplier_np);
    # d_multiplier_dphi[k] = sum_{i=1}^{k+1} i*phi**(i-1) (mirrors
    # _forecast_multiplier_grad_np), built incrementally in one pass.
    cdef np.ndarray[np.float64_t, ndim=1] multiplier = np.empty(h)
    cdef np.ndarray[np.float64_t, ndim=1] d_multiplier_dphi = np.empty(h)
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

    # forward + loss dispatch: forecasts[t, k] = levels[t] + multiplier[k] * trends[t],
    # actuals[t, k] = y[t+k]. Mirrors _loss_value_np/_loss_grad_np's per-name formulas.
    cdef np.ndarray[np.float64_t, ndim=2] d_forecast = np.empty((n_origins, h))
    cdef double n = n_origins * h
    cdef double loss = 0.0
    cdef double resid, forecast, true, abs_r, quad, lin, e, g
    for t in range(n_origins):
        for k in range(h):
            forecast = levels[t] + multiplier[k] * trends[t]
            true = y[t + k]
            resid = forecast - true
            if loss_id == 0:  # mse
                loss += resid * resid
                g = 2.0 * resid
            elif loss_id == 1:  # mae
                loss += fabs(resid)
                g = 1.0 if resid > 0 else (-1.0 if resid < 0 else 0.0)
            elif loss_id == 2:  # huber
                abs_r = fabs(resid)
                quad = fmin(abs_r, delta)
                lin = abs_r - quad
                loss += 0.5 * quad * quad + delta * lin
                g = fmin(fmax(resid, -delta), delta)
            else:  # pinball
                e = true - forecast
                loss += fmax(tau * e, (tau - 1.0) * e)
                g = -tau if true >= forecast else (1.0 - tau)
            d_forecast[t, k] = g
    loss /= n
    for t in range(n_origins):
        for k in range(h):
            d_forecast[t, k] /= n

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
