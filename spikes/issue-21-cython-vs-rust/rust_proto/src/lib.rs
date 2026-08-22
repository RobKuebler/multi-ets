//! Rust/PyO3 port of holt.py's undamped-Holt, mse-loss objective+gradient (issue #21 spike).
//!
//! Mirrors _holt_forward_np + _states_bptt_grad_np + _holt_objective_and_grad from
//! src/multi_ets/holt.py, specialized to kind="undamped" (phi fixed at 1.0, so no
//! d_phi term) and loss="mse", with no padding mask (n_valid == len(y) always in
//! the current codebase, so the mask is always all-True -- see the ADR).
//!
//! params = [alpha, beta, level0, trend0]; returns (loss, grad) with grad in the
//! same 4-element layout.
//!
//! `y` is taken as a `PyReadonlyArray1` (the `numpy` crate) rather than a `Vec<f64>`:
//! the latter forces PyO3 to copy the whole series out of the NumPy array into a
//! fresh Rust Vec on every call, which is called once per L-BFGS-B iteration --
//! for the "one long fit" shape (n=700) that copy was the entire gap vs. Cython's
//! zero-copy typed memoryview (see the ADR's follow-up numbers). `params` stays a
//! `Vec<f64>`: it's 4 floats, copying it is noise either way.

use numpy::PyReadonlyArray1;
use pyo3::prelude::*;

#[pyfunction]
fn objective_and_grad(params: Vec<f64>, y: PyReadonlyArray1<f64>, h: usize) -> (f64, Vec<f64>) {
    let alpha = params[0];
    let beta = params[1];
    let level0 = params[2];
    let trend0 = params[3];
    let y = y.as_slice().expect("y must be a contiguous, non-negative-strided array");
    let t_max = y.len();
    let n_origins = t_max - h + 1;

    let mut levels = vec![0.0; t_max + 1];
    let mut trends = vec![0.0; t_max + 1];
    levels[0] = level0;
    trends[0] = trend0;
    // SAFETY (here and below): every index used is in 0..=t_max / 0..n_origins
    // / 0..h by construction of the loop range, so bounds checks are
    // provably redundant -- this matches the .pyx's `boundscheck=False`
    // directive, kept for a fair apples-to-apples comparison (ponytail: skips
    // Rust's default bounds checks in these hot loops; revisit if this loop
    // is ever extended to take runtime-variable indices).
    unsafe {
        for t in 0..t_max {
            let level = *levels.get_unchecked(t);
            let trend = *trends.get_unchecked(t);
            let new_level = alpha * y.get_unchecked(t) + (1.0 - alpha) * (level + trend);
            let new_trend = beta * (new_level - level) + (1.0 - beta) * trend;
            *levels.get_unchecked_mut(t + 1) = new_level;
            *trends.get_unchecked_mut(t + 1) = new_trend;
        }
    }

    // Flat (n_origins * h) buffer instead of Vec<Vec<f64>>: the nested version
    // was ~n_origins separate heap allocations (one per origin row), which
    // dwarfed everything else for the "one long fit" shape (n_origins ~= 656)
    // -- this flat layout is what closed most of the remaining gap vs. Cython's
    // single contiguous 2D numpy buffer (see the ADR's follow-up numbers).
    let n = (n_origins * h) as f64;
    let mut d_forecast = vec![0.0; n_origins * h];
    let mut loss = 0.0;
    unsafe {
        for t in 0..n_origins {
            for k in 0..h {
                let forecast = levels.get_unchecked(t) + (k + 1) as f64 * trends.get_unchecked(t);
                let resid = forecast - y.get_unchecked(t + k);
                loss += resid * resid;
                *d_forecast.get_unchecked_mut(t * h + k) = 2.0 * resid / n;
            }
        }
    }
    loss /= n;

    let mut gl_direct = vec![0.0; n_origins];
    let mut gt_direct = vec![0.0; n_origins];
    unsafe {
        for t in 0..n_origins {
            let mut gl_sum = 0.0;
            let mut gt_sum = 0.0;
            for k in 0..h {
                let d = *d_forecast.get_unchecked(t * h + k);
                gl_sum += d;
                gt_sum += (k + 1) as f64 * d;
            }
            *gl_direct.get_unchecked_mut(t) = gl_sum;
            *gt_direct.get_unchecked_mut(t) = gt_sum;
        }
    }

    let mut d_alpha = 0.0;
    let mut d_beta = 0.0;
    let mut gl_next = 0.0;
    let mut gt_next = 0.0;
    unsafe {
        for t in (1..n_origins).rev() {
            let gl_t = gl_direct.get_unchecked(t) + gl_next;
            let gt_t = gt_direct.get_unchecked(t) + gt_next;
            let level_prev = *levels.get_unchecked(t - 1);
            let trend_prev = *trends.get_unchecked(t - 1);
            let y_prev = *y.get_unchecked(t - 1);
            let pre_activation = y_prev - level_prev - trend_prev;

            d_alpha += (gl_t + beta * gt_t) * pre_activation;
            d_beta += gt_t * (levels.get_unchecked(t) - level_prev - trend_prev);

            gl_next = gl_t * (1.0 - alpha) - gt_t * alpha * beta;
            gt_next = gl_t * (1.0 - alpha) + gt_t * (1.0 - alpha * beta);
        }
    }

    let d_level0 = gl_direct[0] + gl_next;
    let d_trend0 = gt_direct[0] + gt_next;

    (loss, vec![d_alpha, d_beta, d_level0, d_trend0])
}

#[pymodule]
fn holt_hot_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(objective_and_grad, m)?)?;
    Ok(())
}
