# 0003: Batch (joint multi-step) fit vs. traditional one-step-ahead fit

## Status

Accepted. Batch (joint multi-step) fitting stays the only fitting strategy
`multi_ets` supports; the one-step-ahead adapter built for this spike is
kept in `benchmarks/` as a reference baseline, not promoted into
`src/multi_ets`.

## Context

`multi_ets`'s stated premise (see CONTEXT.md's "Multi-step joint loss"
entry) is that fitting parameters against a joint loss over every horizon
step `1..h` at once beats the classical approach of fitting one-step-ahead
residuals and letting the recursion extrapolate. The existing
matched-structure comparison (`benchmarks/results/summary.csv`, produced by
`run_comparison.py`) already showed this premise was shaky in places --
`multi_ets` loses to statsmodels' classical fit on `real_gdp` at h=6/h=12
and on `air_passengers` at h=12 -- but that comparison never isolated the
fitting *strategy* on its own: `multi_ets` only ever fits jointly, so there
was nothing to compare it against inside the library itself.

This spike adds that missing baseline: a `multi_ets` model fit exactly like
classical exponential smoothing (one-step-ahead SSE, no joint horizon loss),
then forecast by letting the fitted recursion extrapolate forward -- the
same thing statsmodels' `ExponentialSmoothing` does, implemented with
`multi_ets`'s own classes.

### How the one-step-ahead adapter works

`Holt`/`HoltWinters` already separate "what the loss was computed over" from
"what `predict()` forecasts": `fit()` stores only `params_` and the training
series, no `h`-dependent state. So a one-step-ahead fit is just:

1. Construct the model with `h=1` -- the joint loss over horizon `1..1`
   *is* plain one-step-ahead SSE, matching the classical objective exactly.
2. Fit on the training data.
3. Before calling `predict()`, set `model.h` to the desired forecast
   horizon. `predict()` reads `self.h` fresh each call and extrapolates the
   fitted level/trend/season deterministically (Hyndman & Athanasopoulos'
   standard multi-step formula) -- there's no new data to feed the
   recursion, so this is the same computation classical ETS uses to
   forecast past step 1.

This is `benchmarks.adapters.one_step_ahead_adapter()`. It reuses
`_multi_ets_model()` unchanged; no changes to `src/multi_ets` were needed.

### Benchmark setup

`benchmarks/run_model_comparison.py` runs `backtest_split()` (rolling-origin
CV, `benchmarks/evaluation.py`) for every benchmark series and every valid
horizon in `HORIZONS = (1, 6, 12)`, across four methods: `multi_ets`
batch-fit (`n_starts=10`), the new `multi_ets` one-step-ahead fit, and
statsmodels/statsforecast as external references. Full results:
`benchmarks/results/model_comparison.csv` (gitignored; regenerate with
`uv run python -m benchmarks.run_model_comparison`).

At h=1 the joint loss reduces to the same one-step SSE the one-step-ahead
adapter fits, so those two columns are identical there by construction --
all comparisons below use h=6 and h=12, where they diverge.

## Comparison (h=6, h=12; 27 series/horizon combinations)

| Comparison | Win rate | Median MSE ratio | Mean MSE ratio |
|---|---|---|---|
| batch vs. one-step-ahead | batch wins 81% (22/27) | 0.973 | 0.975 |
| batch vs. statsmodels | batch wins 70% (19/27) | 0.989 | 1.031 |

By MASE (scale-free), batch wins 12/27, statsforecast 9/27, statsmodels
5/27, one-step-ahead 1/27.

**Fitting cost:** one-step-ahead fitting is modestly cheaper (median fit
time 0.059s vs. batch's 0.077s across the full sweep) but not dramatically
so -- both run the same `L-BFGS-B` multi-start search; the one-step
objective just has a shorter forecast window to sum over per iteration.
Predict time is identical between the two (same extrapolation formula).

**Where batch loses badly:** `real_gdp` (h=6: 12% worse, h=12: 26% worse)
and `air_passengers`/`synthetic_short_series` at h=6 (13-19% worse). These
are exactly the series that motivated this spike -- the joint multi-step
optimizer is landing in a worse basin for these specific structures, not a
universal failure mode.

**Where batch wins big:** `air_passengers` h=12 (20% better), `nile_flow`
h=6 (15% better), and every `damped` trend structure at h=6 (13-15%
better) -- damped trend is where letting the loss see the full horizon most
plausibly helps, since naive one-step recursion has no signal about how the
damping should behave multiple steps out.

## Recommendation

**Keep batch-only fitting; do not add a one-step-ahead fitting mode.**

The joint multi-step loss beats the naive one-step-ahead-then-recurse
approach on both count (81% of h=6/h=12 cases) and central tendency (2.7%
lower median MSE), and also edges out statsmodels' classical fit on the
same basis. The premise isn't *uniformly* true -- `real_gdp` is a clear,
repeatable counterexample, and it's worth understanding why before
dismissing it -- but "not universally better" is a different, weaker claim
than "doesn't really show that", and the aggregate evidence favors keeping
the current approach over adding a second fitting mode for a marginal,
inconsistent gain.

If `real_gdp`-like regressions turn out to matter in practice (e.g. a
production series behaves like it), the fallback isn't a new fitting mode:
`select_model()` (#12) already CV-selects between structures, and could be
extended to CV-select fitting strategy too. That's a bigger change than
this spike's scope and isn't proposed here.

## Consequences

- `multi_ets` continues to fit only via the joint multi-step loss; no
  fitting-mode option is added to `Holt`/`HoltWinters`'s public API.
- `benchmarks/adapters.py::one_step_ahead_adapter()` and
  `benchmarks/run_model_comparison.py` stay in the benchmark suite as the
  reference baseline this decision was made against, so a future series that
  regresses under batch fitting can be checked against it without
  re-deriving the adapter.
- Revisit if a real-world series behaves like `real_gdp` (batch fit clearly
  worse at longer horizons) -- the documented next step is extending
  `select_model()` (#12) to CV-select fitting strategy, not reopening this
  ADR's recommendation from scratch.
