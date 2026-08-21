"""Matplotlib output: one forecast-overlay plot per series, plus two summary charts."""

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from benchmarks.adapters import Adapter

LIBRARY_COLORS = {
    "multi_ets": "tab:blue",
    "statsmodels": "tab:orange",
    "statsforecast": "tab:green",
}


def plot_series_forecast(
    name: str, y: np.ndarray, h: int, adapters: dict[str, Adapter], out_dir: Path
) -> None:
    """Fit every adapter on a single last-h holdout and plot actuals + all forecasts."""
    train, test = y[:-h], y[-h:]
    t_train = np.arange(len(train))
    t_test = np.arange(len(train), len(train) + h)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(t_train, train, color="black", lw=1, label="actual (train)")
    ax.plot(t_test, test, color="black", lw=1, ls="--", label="actual (holdout)")
    for library, fit_predict in adapters.items():
        try:
            pred = fit_predict(train, h)
        except Exception as exc:
            print(f"  [plot] {name}: {library} failed ({exc}); skipping in plot")
            continue
        ax.plot(t_test, pred, color=LIBRARY_COLORS[library], marker="o", ms=3, label=library)

    ax.set_title(name)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / f"series_{name}.png", dpi=110)
    plt.close(fig)


def plot_relative_mse_boxplot(df: pd.DataFrame, out_dir: Path) -> None:
    """Per (series, h), MSE relative to the best model there -- comparable across series scales."""
    matched = df[df["section"] == "matched"].copy()
    matched["relative_mse"] = matched.groupby(["series", "h"])["mse"].transform(
        lambda s: s / s.min()
    )

    fig, ax = plt.subplots(figsize=(6, 4))
    data = [matched.loc[matched["library"] == lib, "relative_mse"] for lib in LIBRARY_COLORS]
    ax.boxplot(data, tick_labels=list(LIBRARY_COLORS))
    ax.axhline(1.0, color="gray", lw=0.8, ls=":")
    ax.set_ylabel("MSE relative to best model (1.0 = best)")
    ax.set_title("Matched-structure comparison: relative forecast MSE")
    fig.tight_layout()
    fig.savefig(out_dir / "summary_relative_mse.png", dpi=110)
    plt.close(fig)


def plot_timing_bar(df: pd.DataFrame, out_dir: Path) -> None:
    matched = df[df["section"] == "matched"]
    medians = matched.groupby("library")["fit_time"].median()

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(medians.index, medians.values, color=[LIBRARY_COLORS[lib] for lib in medians.index])
    ax.set_ylabel("median fit+predict time per CV fold (s)")
    ax.set_yscale("log")
    ax.set_title("Matched-structure comparison: warm-loop speed")
    fig.tight_layout()
    fig.savefig(out_dir / "summary_timing.png", dpi=110)
    plt.close(fig)
