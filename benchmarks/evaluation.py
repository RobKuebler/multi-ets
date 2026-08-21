"""Rolling-origin backtest (MSE) + a small JIT cold/warm timing footnote for multi_ets."""

import time
from dataclasses import dataclass

import numpy as np

from benchmarks.adapters import Adapter, _multi_ets_model
from benchmarks.datasets import Structure, make_synthetic

N_SPLITS = 3
HORIZONS = (1, 6, 12)


def valid_horizons(length: int, seasonal_period: int | None, n_splits: int = N_SPLITS) -> list[int]:
    """Horizons from HORIZONS that fit within 25% of the series and leave every fold long enough."""
    cap = max(1, length // 4)
    valid = []
    for h in HORIZONS:
        if h > cap:
            continue
        min_len = n_splits * h + max(h + 1, 2 * seasonal_period if seasonal_period else 0)
        if length >= min_len:
            valid.append(h)
    return valid


@dataclass
class BacktestResult:
    mse: float
    fold_mses: list[float]
    fold_times: list[float]


def backtest(
    fit_predict: Adapter, y: np.ndarray, h: int, n_splits: int = N_SPLITS
) -> BacktestResult:
    """Expanding-window, non-overlapping rolling-origin CV -- mirrors selection.py::select_model."""
    y = np.asarray(y, dtype=float)
    fold_mses, fold_times = [], []
    for split in range(n_splits):
        cutoff = len(y) - (n_splits - split) * h
        train, test = y[:cutoff], y[cutoff : cutoff + h]
        t0 = time.perf_counter()
        pred = fit_predict(train, h)
        fold_times.append(time.perf_counter() - t0)
        fold_mses.append(float(np.mean((np.asarray(pred) - test) ** 2)))
    return BacktestResult(mse=float(np.mean(fold_mses)), fold_mses=fold_mses, fold_times=fold_times)


def multi_ets_jit_overhead(
    structure: Structure, h: int, length: int = 150, n_reps: int = 5, seed: int = 0
) -> tuple[float, float]:
    """(cold_fit_time, warm_median_fit_time) for one (structure, h): isolates JAX's one-time
    JIT compile cost from steady-state dispatch, on a fixed-shape series so the JIT cache
    actually gets reused across the repeats (mirrors examples/benchmark.py::benchmark_fit_speed).
    """
    y = make_synthetic(structure, length=length, seed=seed)

    t0 = time.perf_counter()
    _multi_ets_model(structure, h).fit(y)
    cold = time.perf_counter() - t0

    warm_times = []
    for _ in range(n_reps):
        t0 = time.perf_counter()
        _multi_ets_model(structure, h).fit(y)
        warm_times.append(time.perf_counter() - t0)
    return cold, float(np.median(warm_times))
