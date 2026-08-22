"""Issue #21 spike: NumPy baseline vs. Cython vs. Rust/PyO3 for the fitting hot loop.

Benchmarks the undamped-Holt, mse-loss objective+gradient (see cython_proto/holt_hot.pyx
and rust_proto/src/lib.rs for what was ported and why) on the two shapes named in the
ticket. Run from the repo root: uv run python spikes/issue-21-cython-vs-rust/bench.py
(build the extensions first -- see README.md in this directory).
"""

import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).parent / "cython_proto"))
import holt_hot
import holt_hot_rs

from multi_ets.holt import Holt, _holt_objective_and_grad

N_TRIALS_MANY = 5  # repeats of the "many tiny fits" shape
N_FITS_PER_TRIAL = 200  # ticket says "~1000 fits in a loop"; scaled down 5x for spike turnaround,
# per-fit overhead ratios don't change with the loop's length -- see the ADR
N_TRIALS_LONG = 15  # repeats of the "one long fit" shape


def make_series(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.cumsum(rng.normal(0, 1, n)) + 10.0


def fit_numpy(y: np.ndarray, h: int, n_starts: int) -> np.ndarray:
    return Holt(h=h, loss="mse", n_starts=n_starts).fit(y).params_


def _multistart(objective_and_grad, y, h, n_starts) -> np.ndarray:
    """Same multistart loop as Holt.fit(), but calling a compiled objective+grad."""
    from multi_ets.holt import _BOUND_EPS, _heuristic_init, _start_grid

    free_guess = _heuristic_init(y)
    bounds = [
        (_BOUND_EPS, 1 - _BOUND_EPS),
        (_BOUND_EPS, 1 - _BOUND_EPS),
        (None, None),
        (None, None),
    ]
    best = None
    for start in _start_grid(n_starts, 2):
        x0 = np.concatenate([start, free_guess])
        result = minimize(
            lambda p: objective_and_grad(p, y, h),
            x0,
            jac=True,
            method="L-BFGS-B",
            bounds=bounds,
        )
        if best is None or result.fun < best.fun:
            best = result
    return best.x


def fit_cython(y: np.ndarray, h: int, n_starts: int) -> np.ndarray:
    return _multistart(holt_hot.objective_and_grad, y, h, n_starts)


def fit_rust(y: np.ndarray, h: int, n_starts: int) -> np.ndarray:
    def obj(p, y_, h_):
        # y_ passed straight through as a numpy array -- PyReadonlyArray1 on the
        # Rust side borrows its buffer with no copy, unlike the list(y_) this
        # used to do (see the "how is cython so much faster" follow-up in the ADR).
        loss, grad = holt_hot_rs.objective_and_grad(list(p), y_, h_)
        return loss, np.array(grad)

    return _multistart(obj, y, h, n_starts)


def fit_statsmodels(y: np.ndarray, h: int, n_starts: int) -> None:
    from statsmodels.tsa.holtwinters import Holt as SMHolt

    SMHolt(y, initialization_method="estimated", damped_trend=False).fit(optimized=True)


def fit_statsforecast(y: np.ndarray, h: int, n_starts: int) -> None:
    from statsforecast.models import Holt as SFHolt

    SFHolt().fit(y)


def time_it(fn, *args, n_reps: int) -> tuple[float, float]:
    """(first_call_s, min_of_remaining_s). min, not median: this is a shared desktop

    machine (browser + other processes contending for CPU), and min is far less
    sensitive to contention pauses than median for a handful of short reps.
    """
    t0 = time.perf_counter()
    fn(*args)
    first = time.perf_counter() - t0
    reps = []
    for _ in range(n_reps - 1):
        t0 = time.perf_counter()
        fn(*args)
        reps.append(time.perf_counter() - t0)
    return first, min(reps) if reps else first


def bench_many_tiny(impl_name: str, fit_fn) -> None:
    def run_many():
        for i in range(N_FITS_PER_TRIAL):
            y = make_series(40, seed=i)
            fit_fn(y, h=3, n_starts=5)

    first, best = time_it(run_many, n_reps=N_TRIALS_MANY)
    print(
        f"{impl_name:13s} many-tiny ({N_FITS_PER_TRIAL} fits, n=40, h=3, n_starts=5): "
        f"first={first:7.3f}s  min={best:7.3f}s  "
        f"({1000 * best / N_FITS_PER_TRIAL:.3f} ms/fit)"
    )


def bench_one_long(impl_name: str, fit_fn) -> None:
    y = make_series(700, seed=0)

    def run_one():
        fit_fn(y, h=45, n_starts=10)

    first, best = time_it(run_one, n_reps=N_TRIALS_LONG)
    print(
        f"{impl_name:13s} one-long  (n=700, h=45, n_starts=10): "
        f"first={first:7.3f}s  min={best:7.3f}s"
    )


def sanity_check() -> None:
    y = make_series(40, seed=0)
    h = 3
    params = np.array([0.3, 0.2, y[0], 0.1])
    loss_np, grad_np = _holt_objective_and_grad(params, y, h, "undamped", ("mse", 1.0, 0.5))
    loss_cy, grad_cy = holt_hot.objective_and_grad(params, y, h)
    loss_rs, grad_rs = holt_hot_rs.objective_and_grad(list(params), y, h)
    np.testing.assert_allclose(loss_cy, loss_np, atol=1e-10)
    np.testing.assert_allclose(grad_cy, grad_np, atol=1e-10)
    np.testing.assert_allclose(loss_rs, loss_np, atol=1e-10)
    np.testing.assert_allclose(np.array(grad_rs), grad_np, atol=1e-10)
    print("sanity check: cython and rust objective+grad match the numpy baseline (atol=1e-10)\n")


IMPLS = [
    ("numpy", fit_numpy),
    ("cython", fit_cython),
    ("rust", fit_rust),
    ("statsmodels", fit_statsmodels),
    ("statsforecast", fit_statsforecast),
]

if __name__ == "__main__":
    sanity_check()
    print(
        "note: statsmodels/statsforecast fit a single classical 1-step-SSE model,\n"
        "no multistart -- included as a wall-clock speed reference only, not an\n"
        "apples-to-apples comparison with multi_ets's multi-step joint loss (see\n"
        'CONTEXT.md\'s "Matched-structure comparison" entry).\n'
    )
    for name, fn in IMPLS:
        bench_many_tiny(name, fn)
    print()
    for name, fn in IMPLS:
        bench_one_long(name, fn)
