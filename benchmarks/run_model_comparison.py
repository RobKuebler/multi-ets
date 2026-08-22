"""Long-format CV comparison: multi_ets (n_starts=1,5,10) vs. statsmodels vs.
statsforecast, across every valid CV horizon for every benchmark series --
fit time, predict time, and CV MSE/MAPE/MASE, one row per (series, h, method),
plus each series' seasonal period and length.

Run with: uv run python -m benchmarks.run_model_comparison
"""

from pathlib import Path

import pandas as pd

from benchmarks.adapters import FitPredictAdapter, one_step_ahead_adapter, split_adapters
from benchmarks.datasets import Structure, all_series
from benchmarks.evaluation import backtest_split, backtest_split_metrics, valid_horizons

RESULTS_DIR = Path(__file__).parent / "results"
N_STARTS_VALUES = (1, 5, 10)


def _methods(structure: Structure) -> dict[str, FitPredictAdapter]:
    methods = {
        f"multi_ets (n_starts={n})": split_adapters(structure, n_starts=n)["multi_ets"]
        for n in N_STARTS_VALUES
    }
    methods["multi_ets (one-step-ahead)"] = one_step_ahead_adapter(structure, n_starts=10)
    methods.update(
        (library, adapter)
        for library, adapter in split_adapters(structure).items()
        if library != "multi_ets"
    )
    return methods


def run() -> pd.DataFrame:
    rows = []
    for series in all_series():
        h_list = valid_horizons(len(series.y), series.structure.seasonal_period)
        if not h_list:
            print(f"[skip] {series.name}: too short for any tested horizon")
            continue

        methods = _methods(series.structure)
        for h in h_list:
            for method, adapter in methods.items():
                try:
                    result = backtest_split(adapter, series.y, h)
                except Exception as exc:
                    print(f"[skip] {series.name} h={h} {method}: {exc}")
                    continue
                metrics = backtest_split_metrics(result, series.structure.seasonal_period)
                rows.append(
                    {
                        "series": series.name,
                        "length": len(series.y),
                        "seasonal_period": series.structure.seasonal_period or 1,
                        "h": h,
                        "method": method,
                        **metrics,
                    }
                )
        print(f"[done] {series.name} (h={h_list})")

    return pd.DataFrame(rows)


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    df = run()
    df.to_csv(RESULTS_DIR / "model_comparison.csv", index=False)
    print()
    print(df.to_string(index=False))
    print(f"\nFull results written to {RESULTS_DIR}/model_comparison.csv")


if __name__ == "__main__":
    main()
