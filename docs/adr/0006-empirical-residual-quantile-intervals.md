# 0006: Prediction intervals from empirical per-step residual quantiles

## Status

Accepted.

## Context

`predict_interval` used to build a Gaussian interval per horizon step:
`point +/- z * np.nanstd(resid_k)`, where `resid_k` is actual - forecast at
step `k` across every training origin. `nanstd` subtracts the residuals'
own mean, so the interval was always centered on the point forecast even
when the fit is biased. That bias is often deliberate here: a `pinball`
or `expectile` fit with `tau != 0.5`, or a joint multi-step fit that
trades step-1 accuracy for later steps, both leave residuals that are not
centered on zero. A symmetric interval around the point also ignores skew.

Intervals also need to exist for horizon steps the model was not trained on
(an `h` set larger after `fit()`, see ADR 0003, or the gaps of a list
`h`), and for in-sample points, which the sktime adapter needs.

## Decision

Intervals and quantiles come from **empirical quantiles of the step-k
in-sample residuals**, added to the point forecast:
`lower = point + q_{(1-c)/2}(resid_k)`, `upper = point + q_{(1+c)/2}(resid_k)`.
Bias and skew of the fit carry into the interval as they are.

- Residuals come from one forward pass with the fitted params at horizon
  `max(steps)`, over `y` padded with trailing NaNs so every origin counts
  for every step. Any step `k` is covered, not just the training steps.
  NaN actuals (gaps, the pad) are ignored.
- A step with fewer than `_MIN_RESIDUALS = 30` residuals falls back to a
  Gaussian that keeps the mean error, `point + mean + z * std`, with one
  `UserWarning` per call. A step with no residual at all (longer than the
  series) is NaN.
- `predict_quantiles(alphas)` is the one implementation; `predict_interval`
  is a thin wrapper over it, and `predict_in_sample(step, alphas=...)` adds
  the same step-`k` residual quantiles to the in-sample predictions.

### Rejected alternatives

- **Gaussian with per-step `nanstd`** (the old method): centers the
  residuals, so a biased fit gets an interval around the wrong value, and
  it forces symmetry.
- **Loss-matched distributional intervals** (e.g. Poisson/negbinom/Tweedie
  quantiles around the fitted mean, Gaussian for mse, Laplace for mae):
  only defined for mse/mae/poisson/negbinom/tweedie. huber, logcosh,
  expectile and custom callable losses have no matching distribution, so
  this would need a second method for half the losses anyway.
- **Analytical ETS variance formulas** (Hyndman et al.): they assume an
  innovations model fit on one-step residuals, with step-k variance derived
  from the one-step error variance. A joint multi-step fit is neither (see
  CONTEXT.md's "Innovations ETS" and ADR 0003), so the formulas do not
  describe the errors this library actually makes.
- **Built-in conformal intervals**: sktime's `ConformalIntervals` wrapper
  already does this on top of any forecaster, including the multi-ets
  adapter, so it does not need to live in the core.

## Consequences

- In-sample residuals are optimistic: the params were fit to make exactly
  these errors small, so intervals tend to under-cover out of sample,
  more so on short series and for flexible (seasonal, many-param) models.
  Use sktime's `ConformalIntervals` for calibrated out-of-sample coverage.
- Intervals are no longer guaranteed to contain the point forecast; a
  strongly biased fit gives an interval on one side of it, by design.
- Short series (fewer than 30 residuals per step) get a warning and the
  Gaussian fallback rather than a hard error.
