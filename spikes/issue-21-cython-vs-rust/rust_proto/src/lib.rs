//! Rust/PyO3 port of holt.py's undamped-Holt, mse-loss objective+gradient (issue #21 spike).
//!
//! Mirrors _holt_forward_np + _states_bptt_grad_np + _holt_objective_and_grad from
//! src/multi_ets/holt.py, specialized to kind="undamped" (phi fixed at 1.0, so no
//! d_phi term) and loss="mse", with no padding mask (n_valid == len(y) always in
//! the current codebase, so the mask is always all-True -- see the ADR).
//!
//! params = [alpha, beta, level0, trend0]; returns (loss, grad) with grad in the
//! same 4-element layout.

use pyo3::prelude::*;

#[pyfunction]
fn objective_and_grad(params: Vec<f64>, y: Vec<f64>, h: usize) -> (f64, Vec<f64>) {
    let alpha = params[0];
    let beta = params[1];
    let level0 = params[2];
    let trend0 = params[3];
    let t_max = y.len();
    let n_origins = t_max - h + 1;

    let mut levels = vec![0.0; t_max + 1];
    let mut trends = vec![0.0; t_max + 1];
    levels[0] = level0;
    trends[0] = trend0;
    for t in 0..t_max {
        let level = levels[t];
        let trend = trends[t];
        let new_level = alpha * y[t] + (1.0 - alpha) * (level + trend);
        let new_trend = beta * (new_level - level) + (1.0 - beta) * trend;
        levels[t + 1] = new_level;
        trends[t + 1] = new_trend;
    }

    let n = (n_origins * h) as f64;
    let mut d_forecast = vec![vec![0.0; h]; n_origins];
    let mut loss = 0.0;
    for t in 0..n_origins {
        for k in 0..h {
            let forecast = levels[t] + (k + 1) as f64 * trends[t];
            let resid = forecast - y[t + k];
            loss += resid * resid;
            d_forecast[t][k] = 2.0 * resid / n;
        }
    }
    loss /= n;

    let mut gl_direct = vec![0.0; n_origins];
    let mut gt_direct = vec![0.0; n_origins];
    for t in 0..n_origins {
        let mut gl_sum = 0.0;
        let mut gt_sum = 0.0;
        for k in 0..h {
            gl_sum += d_forecast[t][k];
            gt_sum += (k + 1) as f64 * d_forecast[t][k];
        }
        gl_direct[t] = gl_sum;
        gt_direct[t] = gt_sum;
    }

    let mut d_alpha = 0.0;
    let mut d_beta = 0.0;
    let mut gl_next = 0.0;
    let mut gt_next = 0.0;
    for t in (1..n_origins).rev() {
        let gl_t = gl_direct[t] + gl_next;
        let gt_t = gt_direct[t] + gt_next;
        let level_prev = levels[t - 1];
        let trend_prev = trends[t - 1];
        let y_prev = y[t - 1];
        let pre_activation = y_prev - level_prev - trend_prev;

        d_alpha += (gl_t + beta * gt_t) * pre_activation;
        d_beta += gt_t * (levels[t] - level_prev - trend_prev);

        gl_next = gl_t * (1.0 - alpha) - gt_t * alpha * beta;
        gt_next = gl_t * (1.0 - alpha) + gt_t * (1.0 - alpha * beta);
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
