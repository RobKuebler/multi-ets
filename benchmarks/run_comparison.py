"""Benchmark multi_ets against statsmodels and statsforecast: MSE + speed.

Primary comparison: matched trend/seasonal structure across all three libraries,
isolating the effect of multi_ets's multi-step loss vs classical/innovations
1-step fitting (see CONTEXT.md). Bonus comparison: each library's own native
model-selection procedure. Run with: uv run python -m benchmarks.run_comparison
"""

from pathlib import Path

import pandas as pd

from benchmarks.adapters import autoselect_adapters, fitting_method_label, matched_adapters
from benchmarks.datasets import Structure, all_series
from benchmarks.evaluation import backtest, multi_ets_jit_overhead, valid_horizons
from benchmarks.plotting import plot_relative_mse_boxplot, plot_series_forecast, plot_timing_bar

RESULTS_DIR = Path(__file__).parent / "results"

# One representative structure per multi_ets model class, for the JIT footnote.
JIT_FOOTNOTE_STRUCTURES = {
    "SES": Structure("none", "none"),
    "Holt": Structure("add", "none"),
    "DampedHolt": Structure("damped", "none"),
    "HoltWinters": Structure("add", "add", 12),
    "DampedHoltWinters": Structure("damped", "mul", 12),
}


def _structure_label(s: Structure) -> str:
    if s.seasonal == "none":
        return f"trend={s.trend}"
    return f"trend={s.trend}, seasonal={s.seasonal}(m={s.seasonal_period})"


def run() -> pd.DataFrame:
    RESULTS_DIR.mkdir(exist_ok=True)
    rows = []

    for series in all_series():
        h_list = valid_horizons(len(series.y), series.structure.seasonal_period)
        if not h_list:
            print(f"[skip] {series.name}: too short for any tested horizon")
            continue

        matched = matched_adapters(series.structure)
        autosel = autoselect_adapters(series.structure.seasonal_period)

        # Matched section: full rolling-origin CV, every valid horizon -- this
        # is the primary comparison and needs to be rigorous.
        for h in h_list:
            for library, fit_predict in matched.items():
                try:
                    result = backtest(fit_predict, series.y, h)
                except Exception as exc:
                    print(f"[skip] {series.name} h={h} matched/{library}: {exc}")
                    continue
                method = fitting_method_label(library, series.structure)
                for fold, (mse, t) in enumerate(
                    zip(result.fold_mses, result.fold_times, strict=True)
                ):
                    rows.append(
                        {
                            "series": series.name,
                            "category": series.category,
                            "structure": _structure_label(series.structure),
                            "h": h,
                            "section": "matched",
                            "library": library,
                            "fitting_method": method,
                            "fold": fold,
                            "mse": mse,
                            "fit_time": t,
                        }
                    )

        # Auto-selection bonus: single holdout, largest horizon only -- each
        # library's own model search is expensive per call, and this section
        # only needs to show whether auto-selection lands near the matched
        # comparison, not a fully rigorous CV of its own.
        h_auto = h_list[-1]
        for library, fit_predict in autosel.items():
            try:
                result = backtest(fit_predict, series.y, h_auto, n_splits=1)
            except Exception as exc:
                print(f"[skip] {series.name} h={h_auto} autoselect/{library}: {exc}")
                continue
            for fold, (mse, t) in enumerate(zip(result.fold_mses, result.fold_times, strict=True)):
                rows.append(
                    {
                        "series": series.name,
                        "category": series.category,
                        "structure": _structure_label(series.structure),
                        "h": h_auto,
                        "section": "autoselect",
                        "library": library,
                        "fitting_method": "native auto-selection",
                        "fold": fold,
                        "mse": mse,
                        "fit_time": t,
                    }
                )

        print(f"[done] {series.name} (h={h_list})")
        plot_series_forecast(series.name, series.y, max(h_list), matched, RESULTS_DIR)

    return pd.DataFrame(rows)


def run_jit_footnote() -> pd.DataFrame:
    rows = []
    for name, structure in JIT_FOOTNOTE_STRUCTURES.items():
        cold, warm = multi_ets_jit_overhead(structure, h=6)
        rows.append({"multi_ets_class": name, "cold_fit_s": cold, "warm_median_fit_s": warm})
    return pd.DataFrame(rows)


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)

    df = run()
    df.to_csv(RESULTS_DIR / "raw_results.csv", index=False)

    summary = (
        df.groupby(["series", "h", "section", "library", "fitting_method"])
        .agg(mean_mse=("mse", "mean"), median_fit_time=("fit_time", "median"))
        .reset_index()
    )
    summary.to_csv(RESULTS_DIR / "summary.csv", index=False)

    plot_relative_mse_boxplot(df, RESULTS_DIR)
    plot_timing_bar(df, RESULTS_DIR)

    print("\n=== JIT cold-start footnote (multi_ets only; see CONTEXT.md/README) ===")
    jit_df = run_jit_footnote()
    jit_df.to_csv(RESULTS_DIR / "jit_overhead.csv", index=False)
    print(jit_df.to_string(index=False))

    print("\n=== Matched-structure comparison: mean MSE by series/h/library ===")
    matched_summary = summary[summary["section"] == "matched"]
    print(
        matched_summary.pivot_table(
            index=["series", "h"], columns="library", values="mean_mse"
        ).to_string()
    )

    print(f"\nFull results written to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
