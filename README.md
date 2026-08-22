# multi-ets

Exponential smoothing (Holt, Holt-Winters) fit by minimizing a **joint multi-step-ahead loss**, instead of the classical one-step-ahead loss statsmodels and most other implementations use.

Most ETS implementations fit `alpha`/`beta`/`gamma`/... to minimize one-step-ahead residuals, then hope the recursion also forecasts well `h` steps out. Here, the loss is computed jointly across every valid forecast origin and every horizon step `1..h`, so the fitted parameters are the ones that actually minimize the error you care about: the multi-step forecast.

Fit with hand-derived gradients and multi-start [L-BFGS-B](https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html) (via `scipy.optimize.minimize`) to deal with the resulting loss surface's local optima.

## Install

```bash
uv add multi-ets   # or: pip install multi-ets
```

## Usage

```python
from multi_ets import SES, Holt, DampedHolt, HoltWinters, DampedHoltWinters

# Flat forecast, no trend.
forecasts = SES(h=5).fit(y).predict()

# Plain linear trend, forecast 5 steps ahead.
forecasts = Holt(h=5).fit(y).predict()

# Damped trend (phi jointly optimized, box-constrained to (0, 1)).
forecasts = DampedHolt(h=5).fit(y).predict()

# Additive seasonality, period 12 (e.g. monthly data).
forecasts = HoltWinters(h=5, seasonal_period=12).fit(y).predict()

# Multiplicative seasonality, with a damped trend.
forecasts = (
    DampedHoltWinters(h=5, seasonal_period=12, seasonal_type="multiplicative").fit(y).predict()
)
```

Every model exposes just `.fit(y)` / `.predict()`. Fitted parameters are in `.params_` after `fit()`.

### Prediction intervals

Every model exposes `.predict_interval(coverage=0.9)` after fitting, returning an empirical
per-horizon-step Gaussian interval built from residuals of the fitted joint loss:

```python
lower, upper = Holt(h=5).fit(y).predict_interval(coverage=0.9)

# A list of coverages returns a {coverage: (lower, upper)} dict instead.
intervals = Holt(h=5).fit(y).predict_interval(coverage=[0.5, 0.9])
```

### Model selection

`select_model` picks the best of several already-configured candidates via rolling-origin
cross-validation, scoring each with its own `loss`/`h`:

```python
from multi_ets import select_model

candidates = [
    SES(h=5, loss="mae"),
    HoltWinters(h=5, seasonal_period=12, loss="mae"),
]
best, scores = select_model(candidates, y, n_splits=3)
forecasts = best.predict()
```

`best` is refit on the full series; `scores` maps each candidate to its mean CV loss.

### Loss

All models accept a `loss` kwarg: `"mse"` (default), `"mae"`, `"huber"` (+ `delta`), `"pinball"` (+ `tau`), or any callable `(y_true, y_pred) -> scalar` written in plain NumPy.

```python
Holt(h=5, loss="huber", delta=2.0).fit(y).predict()
Holt(h=5, loss=lambda yt, yp: (yt - yp).var()).fit(y).predict()
```

### Multi-start

Fitting is non-convex, so each model restarts from `n_starts` (default 10) points spread across the parameter box and keeps the best, via a plain Python loop over `scipy.optimize.minimize`.

### Fast (optional compiled) backend

`multi_ets` installs and runs correctly as a pure-Python wheel with zero C toolchain. A compiled
Cython hot loop (the fit objective + hand-derived gradient) is available as an opt-in speedup for
built-in losses (`mse`/`mae`/`huber`/`pinball`); a custom callable `loss` always uses the NumPy
path. When the compiled extension isn't built, everything silently falls back to NumPy -- nothing
breaks either way. See `docs/adr/0002-optional-compiled-extension.md` for the full decision.

To build it:

```bash
uv sync --extra fast   # or: pip install -e ".[fast]" -- installs cython + setuptools
cd src/multi_ets && uv run python setup.py build_ext --inplace
```

Note: `uv sync --extra fast` alone does **not** build the extension -- this project's build backend
(`uv_build`) only supports pure-Python packages and has no hook for compiling `.pyx` files as part
of install. The `build_ext --inplace` step above is a real, separate manual step, not optional
boilerplate.

## Model overview

| Class | Trend | Seasonality |
|---|---|---|
| `SES` | — | — |
| `Holt` | linear | — |
| `DampedHolt` | damped (`phi` fit jointly) | — |
| `HoltWinters` | linear | additive or multiplicative (`seasonal_type`) |
| `DampedHoltWinters` | damped | additive or multiplicative |

## Development

```bash
uv run pytest tests/ -q
uv run python examples/benchmark.py   # accuracy/speed vs. statsmodels on real datasets

uv run pre-commit install   # once, to run lint/format/type checks on every commit
uv run pre-commit run --all-files

uv run ruff check .    # lint
uv run ruff format .   # format
uv run ty check src    # type check
```
