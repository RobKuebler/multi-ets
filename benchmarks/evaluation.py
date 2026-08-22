"""Rolling-origin backtest (MSE) for multi_ets, statsmodels, statsforecast."""

import time
from dataclasses import dataclass

import numpy as np

from benchmarks.adapters import Adapter, FitPredictAdapter

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
        if split == 0:
            # statsforecast is numba-jitted: an untimed call here absorbs its
            # one-time compilation cost so fold_times reflects steady-state
            # speed, not a first-call outlier.
            fit_predict(train, h)
        t0 = time.perf_counter()
        pred = np.asarray(fit_predict(train, h))
        fold_times.append(time.perf_counter() - t0)
        fold_mses.append(float(np.mean((pred - test) ** 2)))
    return BacktestResult(mse=float(np.mean(fold_mses)), fold_mses=fold_mses, fold_times=fold_times)


def mape(pred: np.ndarray, test: np.ndarray) -> float:
    """Mean absolute percentage error, in percent; folds with a zero actual are excluded."""
    mask = test != 0
    if not np.any(mask):
        return float("nan")
    return float(np.mean(np.abs((test[mask] - pred[mask]) / test[mask]))) * 100.0


def mase(
    pred: np.ndarray, test: np.ndarray, train: np.ndarray, seasonal_period: int | None
) -> float:
    """Mean absolute error scaled by the in-sample seasonal-naive MAE (Hyndman & Koehler 2006)."""
    m = seasonal_period or 1
    scale = float(np.mean(np.abs(train[m:] - train[:-m])))
    return float(np.mean(np.abs(test - pred))) / scale


@dataclass
class SplitBacktestResult:
    fold_fit_times: list[float]
    fold_predict_times: list[float]
    fold_mses: list[float]
    fold_preds: list[np.ndarray]
    fold_tests: list[np.ndarray]
    fold_trains: list[np.ndarray]


def backtest_split(
    adapter: FitPredictAdapter, y: np.ndarray, h: int, n_splits: int = N_SPLITS
) -> SplitBacktestResult:
    """Like backtest(), but times fit and predict separately."""
    y = np.asarray(y, dtype=float)
    fold_fit_times, fold_predict_times, fold_mses = [], [], []
    fold_preds, fold_tests, fold_trains = [], [], []
    for split in range(n_splits):
        cutoff = len(y) - (n_splits - split) * h
        train, test = y[:cutoff], y[cutoff : cutoff + h]
        if split == 0:
            # statsforecast is numba-jitted: an untimed call here absorbs its
            # one-time compilation cost so fold times reflect steady-state
            # speed, not a first-call outlier.
            adapter.predict(adapter.fit(train, h), h)
        t0 = time.perf_counter()
        fitted = adapter.fit(train, h)
        t1 = time.perf_counter()
        pred = np.asarray(adapter.predict(fitted, h))
        t2 = time.perf_counter()
        fold_fit_times.append(t1 - t0)
        fold_predict_times.append(t2 - t1)
        fold_mses.append(float(np.mean((pred - test) ** 2)))
        fold_preds.append(pred)
        fold_tests.append(test)
        fold_trains.append(train)
    return SplitBacktestResult(
        fold_fit_times=fold_fit_times,
        fold_predict_times=fold_predict_times,
        fold_mses=fold_mses,
        fold_preds=fold_preds,
        fold_tests=fold_tests,
        fold_trains=fold_trains,
    )


def backtest_split_metrics(
    result: SplitBacktestResult, seasonal_period: int | None
) -> dict[str, float]:
    """Median fit/predict time, mean MSE/MAPE/MASE across folds, from a SplitBacktestResult."""
    fold_pairs = zip(result.fold_preds, result.fold_tests, strict=True)
    fold_triples = zip(result.fold_preds, result.fold_tests, result.fold_trains, strict=True)
    return {
        "fit_time_median": float(np.median(result.fold_fit_times)),
        "predict_time_median": float(np.median(result.fold_predict_times)),
        "mse": float(np.mean(result.fold_mses)),
        "mape": float(np.mean([mape(p, t) for p, t in fold_pairs])),
        "mase": float(np.mean([mase(p, t, tr, seasonal_period) for p, t, tr in fold_triples])),
    }
