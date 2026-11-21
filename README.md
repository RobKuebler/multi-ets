# multi-ets

Exponential smoothing (Holt, Holt-Winters) fit by minimizing a **joint multi-step-ahead loss**, instead of the classical one-step-ahead loss statsmodels and most other implementations use.

Most ETS implementations fit `alpha`/`beta`/`gamma`/... to minimize one-step-ahead residuals, then hope the recursion also forecasts well `h` steps out. Here, the loss is computed jointly across every valid forecast origin and every horizon step `1..h`, so the fitted parameters are the ones that actually minimize the error you care about: the multi-step forecast.

Fit with hand-derived gradients and multi-start [L-BFGS-B](https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html) (via `scipy.optimize.minimize`) to deal with the resulting loss surface's local optima.

## Why multi-ets?

statsmodels and statsforecast are solid classical/innovations ETS implementations, but their fitting is rigid: one fixed loss (SSE or Gaussian MLE), and no real answer for missing data. `multi_ets` differs on exactly those two points, and is at least competitive on plain forecast accuracy.

**Missing values.** `multi_ets` skips NaN steps in the state recursion and zero-weights them out of the loss, instead of failing or silently propagating garbage:

```
=== Missing values (2 NaNs in a 20-point series) ===
multi_ets:     [145.00701916 146.36080822 147.71459728]
statsmodels:   [nan nan nan]  (fits without error, but forecast is all-NaN)
statsforecast: raises Exception: no model able to be fitted
```

statsmodels *looks* like it succeeded -- `.fit()` raises nothing -- but the forecast is silently all-NaN. statsforecast fails outright. Neither offers a way to just handle the gap. (`uv run python -m benchmarks.capability_demo`)

**Loss functions.** All 9 built-in losses (`mse`, `mae`, `huber`, `pinball`, `expectile`, `logcosh`, `poisson`, `negbinom`, `tweedie`), or any custom callable, apply to every model. statsmodels and statsforecast don't expose a loss choice at all -- you get SSE (classical) or Gaussian MLE (innovations), period. That matters whenever the default isn't the right objective, e.g. an outlier-robust fit:

```
=== Loss choice on a series with one outlier ===
(ground truth for the first forecast step is ~130.0)
multi_ets (mse         ): [135.01093768 135.62232229 136.2337069  136.84509151 137.45647612]
multi_ets (huber delta=5): [131.34230083 131.87624212 132.41018341 132.9441247  133.47806599]
statsmodels/statsforecast: no loss parameter -- always fit by SSE/MLE.
```

The `huber` fit lands closer to the true continuation because the outlier no longer dominates the loss the way it does under plain `mse`.

**Forecast accuracy and fit time.** On a matched-structure rolling-origin backtest (4 real + 10 synthetic series, `h` in `{1, 6, 12}`, see `docs/adr/0003-batch-vs-one-step-ahead-fitting.md` and `benchmarks/run_model_comparison.py`), the joint multi-step loss beats statsmodels' classical one-step fit on MSE in 70% of h=6/h=12 cases (median MSE ratio 0.989), and comes out ahead by MASE more often than any single alternative (multi_ets 12/27, statsforecast 9/27, statsmodels 5/27). Median fit time is 0.077s vs. statsmodels' 0.120s; statsforecast is fastest at 0.017s (numba-jitted) but that speed comes with no loss choice and no missing-value support. None of this is a landslide in either direction -- `multi_ets` loses on some structures (e.g. `real_gdp` at longer horizons, documented in the ADR) -- but it's a real edge, not just a different API for the same fit.

```bash
uv run python -m benchmarks.run_model_comparison   # regenerate the numbers above
uv run python -m benchmarks.capability_demo         # regenerate the snippets above
```

## Install

```bash
uv add multi-ets   # or: pip install multi-ets
```

## Usage

```python
from multi_ets import ExponentialSmoothing

# Flat forecast, no trend, no seasonality.
forecasts = ExponentialSmoothing(h=5).fit(y).predict()

# Plain linear trend, forecast 5 steps ahead.
forecasts = ExponentialSmoothing(h=5, trend="add").fit(y).predict()

# Damped trend (phi jointly optimized, box-constrained to (0, 1)).
forecasts = ExponentialSmoothing(h=5, trend="damped").fit(y).predict()

# Additive seasonality, period 12 (e.g. monthly data), with a linear trend.
forecasts = ExponentialSmoothing(h=5, trend="add", seasonal_period=12).fit(y).predict()

# Seasonality with no trend term at all.
forecasts = ExponentialSmoothing(h=5, seasonal_period=12).fit(y).predict()

# Multiplicative seasonality, with a damped trend.
forecasts = (
    ExponentialSmoothing(h=5, trend="damped", seasonal_period=12, seasonal_type="multiplicative")
    .fit(y)
    .predict()
)

# Only some horizon steps matter, e.g. days 1-3 and 7-9: trains on horizon 9, but only the
# listed steps count in the loss; predict() returns those 6 values.
forecasts = ExponentialSmoothing(h=[1, 2, 3, 7, 8, 9], trend="add").fit(y).predict()
```

`trend` is `None` (default, no trend), `"add"`, or `"damped"`; `seasonal_period` (default `None`)
adds an `"additive"` or `"multiplicative"` (`seasonal_type`) seasonal component on top of either
trend, or on top of no trend at all. Every combination exposes just `.fit(y)` / `.predict()`;
fitted parameters are in `.params_` after `fit()`.

### Prediction intervals

Every model exposes `.predict_interval(coverage=0.9)` after fitting. Each horizon step `k` gets
the point forecast plus empirical quantiles of the step-`k` in-sample residuals (actual - forecast
over every training origin), so a biased or skewed fit gives an asymmetric interval. With fewer
than 30 residuals for a step it warns and falls back to a Gaussian with the residuals' mean and
std (see `docs/adr/0006`). In-sample residuals are optimistic; for calibrated coverage wrap the
model in sktime's `ConformalIntervals`.

```python
model = ExponentialSmoothing(h=5, trend="add").fit(y)
lower, upper = model.predict_interval(coverage=0.9)

# A list of coverages returns a {coverage: (lower, upper)} dict instead.
intervals = model.predict_interval(coverage=[0.5, 0.9])

# Arbitrary quantiles, shape (n_steps, n_alphas).
quantiles = model.predict_quantiles([0.1, 0.5, 0.9])

# In-sample k-step-ahead predictions (first k - 1 values NaN), optionally with quantiles.
fitted = model.predict_in_sample(step=1)
fitted, quantiles = model.predict_in_sample(step=2, alphas=[0.1, 0.9])
```

### Model selection

`select_model` picks the best of several already-configured candidates via rolling-origin
cross-validation, scoring each with its own `loss`/`h`:

```python
from multi_ets import ExponentialSmoothing, select_model

candidates = [
    ExponentialSmoothing(h=5, loss="mae"),
    ExponentialSmoothing(h=5, trend="add", seasonal_period=12, loss="mae"),
]
best, scores = select_model(candidates, y, n_splits=3)
forecasts = best.predict()
```

`best` is refit on the full series; `scores` maps each candidate to its mean CV loss.

`AutoExponentialSmoothing` builds that candidate grid for you -- every valid `trend` x
`seasonal_type` combination at a given `seasonal_period` (or just the three `trend` options with
no seasonal candidates when `seasonal_period` is omitted) -- and picks the best the same way:

```python
from multi_ets import AutoExponentialSmoothing

model = AutoExponentialSmoothing(h=5, seasonal_period=12, loss="mae").fit(y)
forecasts = model.predict()
model.best_  # the winning fitted ExponentialSmoothing
model.scores_  # {candidate: mean_cv_loss}
```

It's named `AutoExponentialSmoothing` rather than `AutoETS`: `AutoETS` is statsforecast's name for
an *innovations*-based (MLE) auto-selector -- a structurally different fitting philosophy (see
CONTEXT.md's "Innovations ETS" glossary entry) -- and this repo's own benchmarks compare against
it directly.

### sktime integration

```bash
pip install "multi-ets[sktime]"
```

`MultiETSForecaster` and `AutoMultiETSForecaster` wrap the two models as sktime forecasters, with
the same hyperparameters minus `h` (`seasonal_period` is sktime's `sp`). The `fh` passed to
`fit` sets the training horizon: its positive steps are the list `h`, so `fh=[1, 2, 3, 7, 8, 9]`
trains on just those steps. In-sample `fh` points (step <= 0) are `insample_step`-ahead
predictions (default 1). Multivariate or hierarchical `y` is fit one series at a time by sktime.

```python
from sktime.datasets import load_airline
from sktime.forecasting.model_evaluation import evaluate
from sktime.split import ExpandingWindowSplitter

from multi_ets.integrations.sktime import MultiETSForecaster

y = load_airline()  # monthly airline passengers, PeriodIndex

# Train on, and forecast, the next quarter plus months 7-9 only.
forecaster = MultiETSForecaster(
    trend="damped", seasonal_period=12, seasonal_type="multiplicative"
)
forecaster.fit(y, fh=[1, 2, 3, 7, 8, 9])
y_pred = forecaster.predict()
intervals = forecaster.predict_interval(coverage=0.9)

# Rolling-origin backtest, like any other sktime forecaster.
cv = ExpandingWindowSplitter(initial_window=96, step_length=12, fh=[1, 2, 3, 7, 8, 9])
results = evaluate(forecaster, cv=cv, y=y)
print(results[["test_MeanAbsolutePercentageError", "fit_time"]].mean())
```

For calibrated intervals, wrap it in `ConformalIntervals`. Its refits forecast every point after
the training window, so pick an `initial_window` above half the series, and give `y` a
`PeriodIndex` or `DatetimeIndex` (with a plain integer index sktime reads those points as relative
steps):

```python
from sktime.forecasting.conformal import ConformalIntervals

conformal = ConformalIntervals(MultiETSForecaster(trend="add"), initial_window=len(y) * 2 // 3)
intervals = conformal.fit(y, fh=[1, 2, 3]).predict_interval(coverage=0.9)
```

### Loss

All models accept a `loss` kwarg: `"mse"` (default), `"mae"`, `"huber"` (+ `delta`), `"pinball"` (+
`tau`), `"expectile"` (+ `tau`), `"logcosh"`, `"poisson"`, `"negbinom"` (+ `dispersion`), or
`"tweedie"` (+ `p` power, `1 < p < 2`), or any callable `(y_true, y_pred) -> scalar` written in
plain NumPy. `poisson`/`negbinom`/`tweedie` require a non-negative `y`.

```python
ExponentialSmoothing(h=5, trend="add", loss="huber", delta=2.0).fit(y).predict()
ExponentialSmoothing(h=5, trend="add", loss="tweedie", p=1.3).fit(y).predict()
ExponentialSmoothing(h=5, trend="add", loss=lambda yt, yp: (yt - yp).var()).fit(y).predict()
```

### Multi-start

Fitting is non-convex, so each model restarts from `n_starts` (default 10) points spread across the parameter box and keeps the best, via a plain Python loop over `scipy.optimize.minimize`.

### Compiled backend

The fit objective + hand-derived gradient for every built-in loss (`mse`/`mae`/`huber`/`pinball`/
`poisson`/`negbinom`/`tweedie`/`expectile`/`logcosh`) runs in a compiled Cython extension. PyPI ships
prebuilt wheels for Linux, macOS and Windows (x86_64 and ARM), so `pip install multi-ets` gets the
fast version with no compiler needed. A custom callable `loss` always uses the NumPy path.

On a platform without a prebuilt wheel, pip builds from source. If no C compiler is available, the
install still succeeds with a warning and `multi_ets` falls back to the slower NumPy path -- results
are the same either way. See `docs/adr/0005-ship-compiled-wheels.md`.

For development, `uv sync` (or `pip install -e .`) compiles the extension automatically and
recompiles it when `_holt_hot.pyx` changes. The `MULTI_ETS_EXTENSION` environment variable
controls the build: `optional` (default), `required` (fail if compiling fails) or `skip`
(pure Python).

## Model overview

A single `ExponentialSmoothing(h, trend, seasonal_period, seasonal_type, ...)` class covers every
configuration below; the `trend`/`seasonal_period`/`seasonal_type` kwargs pick the configuration
(see `docs/adr/0004-consolidate-model-classes.md` for why this replaced one class per
configuration):

| `trend` | `seasonal_period` | Configuration |
|---|---|---|
| `None` | `None` | flat forecast, no trend, no seasonality (simple exponential smoothing) |
| `"add"` | `None` | linear trend |
| `"damped"` | `None` | damped trend, `phi` fit jointly |
| `None` | set | seasonal only, additive or multiplicative (`seasonal_type`), no trend |
| `"add"` | set | linear trend + seasonal, additive or multiplicative |
| `"damped"` | set | damped trend + seasonal, additive or multiplicative |

## Development

```bash
uv run pytest tests/ -q
uv run python examples/benchmark.py       # accuracy/speed vs. statsmodels on real datasets
uv run python -m benchmarks.capability_demo   # missing-value / loss-choice demo (see "Why multi-ets?")

uv run pre-commit install   # once, to run lint/format/type checks on every commit
uv run pre-commit run --all-files

uv run ruff check .    # lint
uv run ruff format .   # format
uv run ty check src    # type check
```

## License

[MIT No Attribution](LICENSE) (MIT-0): use, modify and redistribute it for any purpose, commercial included, with no attribution required.
