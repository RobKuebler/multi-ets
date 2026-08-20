"""Runnable sanity check for the h-step-optimized Holt model.

Not a rigorous benchmark suite: fits our Holt (loss="mse") against
statsmodels' classical (1-step-optimized) Holt on a few real, publicly
available series with trend structure, and compares multi-step forecast
accuracy and fit time. Run with: uv run python examples/benchmark.py
"""

import time

import numpy as np
import statsmodels.datasets as sm_datasets
from statsmodels.tsa.holtwinters import Holt as SMHolt

from multi_ets import Holt

# Classic Box-Jenkins AirPassengers monthly totals, Jan 1949 - Dec 1960.
AIR_PASSENGERS = np.array(
    [
        112,
        118,
        132,
        129,
        121,
        135,
        148,
        148,
        136,
        119,
        104,
        118,
        115,
        126,
        141,
        135,
        125,
        149,
        170,
        170,
        158,
        133,
        114,
        140,
        145,
        150,
        178,
        163,
        172,
        178,
        199,
        199,
        184,
        162,
        146,
        166,
        171,
        180,
        193,
        181,
        183,
        218,
        230,
        242,
        209,
        191,
        172,
        194,
        196,
        196,
        236,
        235,
        229,
        243,
        264,
        272,
        237,
        211,
        180,
        201,
        204,
        188,
        235,
        227,
        234,
        264,
        302,
        293,
        259,
        229,
        203,
        229,
        242,
        233,
        267,
        269,
        270,
        315,
        364,
        347,
        312,
        274,
        237,
        278,
        284,
        277,
        317,
        313,
        318,
        374,
        413,
        405,
        355,
        306,
        271,
        306,
        315,
        301,
        356,
        348,
        355,
        422,
        465,
        467,
        404,
        347,
        305,
        336,
        340,
        318,
        362,
        348,
        363,
        435,
        491,
        505,
        404,
        359,
        310,
        337,
        360,
        342,
        406,
        396,
        420,
        472,
        548,
        559,
        463,
        407,
        362,
        405,
        417,
        391,
        419,
        461,
        472,
        535,
        622,
        606,
        508,
        461,
        390,
        432,
    ],
    dtype=float,
)


def load_series():
    """Name -> (series, forecast horizon), all publicly available/offline."""
    nile = sm_datasets.nile.load_pandas().data["volume"].to_numpy(dtype=float)
    gdp = sm_datasets.macrodata.load_pandas().data["realgdp"].to_numpy(dtype=float)
    co2 = sm_datasets.co2.load_pandas().data["co2"]
    co2_monthly = co2.resample("MS").mean().interpolate().to_numpy(dtype=float)[-150:]

    return {
        "air_passengers": (AIR_PASSENGERS, 12),
        "nile_flow": (nile, 5),
        "real_gdp": (gdp, 8),
        "co2_monthly": (co2_monthly, 12),
    }


def evaluate(y, h):
    y_train, y_test = y[:-h], y[-h:]

    t0 = time.perf_counter()
    pred_ours = Holt(h=h, loss="mse").fit(y_train).predict()
    t_ours = time.perf_counter() - t0

    t0 = time.perf_counter()
    sm_fit = SMHolt(y_train, initialization_method="estimated", damped_trend=False).fit(
        optimized=True
    )
    pred_sm = sm_fit.forecast(h)
    t_sm = time.perf_counter() - t0

    mse_ours = float(np.mean((pred_ours - y_test) ** 2))
    mse_sm = float(np.mean((pred_sm - y_test) ** 2))
    return mse_ours, mse_sm, t_ours, t_sm


def benchmark_fit_speed(n_reps=5):
    """Repeated .fit() calls on a warm cache: isolates dispatch overhead from
    one-time JIT compilation. Before/after the switch from a sequential
    Python for-loop over n_starts restarts (scipy.optimize.minimize per
    restart) to a single jax.vmap'd multi-start L-BFGS-B solve:

        before (sequential restarts): median ~0.28s  (0.26s-0.31s over 5 reps)
        after  (vmapped restarts):    median ~0.02s  (~13x faster)
    """
    rng = np.random.default_rng(0)
    y = np.cumsum(rng.normal(0.5, 1.0, 100)) + 20
    h = 10

    Holt(h=h, loss="mse", n_starts=10).fit(y)  # warm-up: pay JIT compile once

    times = []
    for _ in range(n_reps):
        t0 = time.perf_counter()
        Holt(h=h, loss="mse", n_starts=10).fit(y)
        times.append(time.perf_counter() - t0)
    return times


def main():
    header = f"{'dataset':<16}{'h':>3}{'MSE ours':>14}{'MSE statsmodels':>18}{'winner':>13}"
    print(header)
    print("-" * len(header))
    for name, (y, h) in load_series().items():
        mse_ours, mse_sm, t_ours, t_sm = evaluate(y, h)
        winner = "ours" if mse_ours < mse_sm else "statsmodels"
        print(
            f"{name:<16}{h:>3}{mse_ours:>14.1f}{mse_sm:>18.1f}{winner:>13}"
            f"   (fit: {t_ours:.2f}s vs {t_sm:.3f}s)"
        )

    print()
    print("fit() speed, 100-point series, h=10, n_starts=10 (warm JIT cache):")
    fit_times = benchmark_fit_speed()
    print("  " + ", ".join(f"{t:.4f}s" for t in fit_times))
    print(f"  median: {sorted(fit_times)[len(fit_times) // 2]:.4f}s")


if __name__ == "__main__":
    main()
