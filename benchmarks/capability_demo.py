"""Runnable demo of two things statsmodels/statsforecast can't do at all:
missing-value handling and loss-function choice. Not a scored benchmark --
see run_comparison.py / run_model_comparison.py for that. Run with:
uv run python -m benchmarks.capability_demo
"""

import numpy as np
from statsforecast.models import AutoETS
from statsmodels.tsa.holtwinters import ExponentialSmoothing as SMExponentialSmoothing

from benchmarks.datasets import AIR_PASSENGERS
from multi_ets import ExponentialSmoothing


def missing_values_demo() -> None:
    y = AIR_PASSENGERS[:20].copy()
    y[5] = np.nan
    y[12] = np.nan

    print("=== Missing values (2 NaNs in a 20-point series) ===")

    forecast = ExponentialSmoothing(h=3, trend="add").fit(y).predict()
    print(f"multi_ets:     {forecast}")

    fit = SMExponentialSmoothing(y, trend="add", initialization_method="estimated").fit(
        optimized=True
    )
    print(f"statsmodels:   {fit.forecast(3)}  (fits without error, but forecast is all-NaN)")

    try:
        AutoETS(season_length=1, model="AAN").fit(y)
        print("statsforecast: fit succeeded")
    except Exception as exc:
        print(f"statsforecast: raises {type(exc).__name__}: {exc}")


def loss_choice_demo() -> None:
    rng = np.random.default_rng(0)
    t = np.arange(60, dtype=float)
    y = 100 + 0.5 * t + rng.normal(0, 2.0, 60)
    y[45] += 80  # one large outlier
    true_next = 100 + 0.5 * 60  # ground-truth level at the first forecast step

    print("\n=== Loss choice on a series with one outlier ===")
    print(f"(ground truth for the first forecast step is ~{true_next:.1f})")
    for loss, kwargs in [("mse", {}), ("huber", {"delta": 5.0})]:
        forecast = ExponentialSmoothing(h=5, trend="add", loss=loss, **kwargs).fit(y).predict()
        print(f"multi_ets ({loss + (' delta=5' if kwargs else ''):<12}): {forecast}")
    print("statsmodels/statsforecast: no loss parameter -- always fit by SSE/MLE.")


if __name__ == "__main__":
    missing_values_demo()
    loss_choice_demo()
