# cython: boundscheck=False, wraparound=False, cdivision=True
"""Cython port of holt.py's undamped-Holt, mse-loss objective+gradient (issue #21 spike).

Mirrors _holt_forward_np + _states_bptt_grad_np + _holt_objective_and_grad from
src/multi_ets/holt.py, specialized to kind="undamped" (phi fixed at 1.0, so no
d_phi term) and loss="mse", with no padding mask (n_valid == len(y) always in
the current codebase, so the mask is always all-True -- see the ADR).

params = [alpha, beta, level0, trend0]; returns (loss, grad) with grad in the
same 4-element layout.
"""
import numpy as np
cimport numpy as np

def objective_and_grad(np.ndarray[np.float64_t, ndim=1] params, np.ndarray[np.float64_t, ndim=1] y, int h):
    cdef double alpha = params[0]
    cdef double beta = params[1]
    cdef double level0 = params[2]
    cdef double trend0 = params[3]
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
        new_level = alpha * y[t] + (1 - alpha) * (level + trend)
        new_trend = beta * (new_level - level) + (1 - beta) * trend
        levels[t + 1] = new_level
        trends[t + 1] = new_trend

    # forward: forecasts[t, k] = levels[t] + (k+1) * trends[t], actuals[t, k] = y[t+k]
    cdef np.ndarray[np.float64_t, ndim=2] d_forecast = np.empty((n_origins, h))
    cdef double n = n_origins * h
    cdef double loss = 0.0
    cdef double resid, forecast
    for t in range(n_origins):
        for k in range(h):
            forecast = levels[t] + (k + 1) * trends[t]
            resid = forecast - y[t + k]
            loss += resid * resid
            d_forecast[t, k] = 2.0 * resid
    loss /= n
    for t in range(n_origins):
        for k in range(h):
            d_forecast[t, k] /= n

    # BPTT, mirroring _states_bptt_grad_np with phi=1.0 fixed (no d_phi term).
    cdef np.ndarray[np.float64_t, ndim=1] gl_direct = np.empty(n_origins)
    cdef np.ndarray[np.float64_t, ndim=1] gt_direct = np.empty(n_origins)
    cdef double gl_sum, gt_sum
    for t in range(n_origins):
        gl_sum = 0.0
        gt_sum = 0.0
        for k in range(h):
            gl_sum += d_forecast[t, k]
            gt_sum += (k + 1) * d_forecast[t, k]
        gl_direct[t] = gl_sum
        gt_direct[t] = gt_sum

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
        pre_activation = y_prev - level_prev - trend_prev

        d_alpha += (gl_t + beta * gt_t) * pre_activation
        d_beta += gt_t * (levels[t] - level_prev - trend_prev)

        gl_next = gl_t * (1 - alpha) - gt_t * alpha * beta
        gt_next = gl_t * (1 - alpha) + gt_t * (1 - alpha * beta)

    cdef double d_level0 = gl_direct[0] + gl_next
    cdef double d_trend0 = gt_direct[0] + gt_next

    cdef np.ndarray[np.float64_t, ndim=1] grad = np.array([d_alpha, d_beta, d_level0, d_trend0])
    return loss, grad
