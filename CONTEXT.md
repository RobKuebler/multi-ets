# multi-ets

A Python library for exponential smoothing (SES, Holt, damped Holt, Holt-Winters) fit by joint multi-step-ahead loss minimization instead of classical one-step-ahead residual fitting.

## Language

**Multi-step joint loss**:
The training objective this library optimizes: a loss computed jointly across every valid forecast origin and every horizon step `1..h`, rather than only the one-step-ahead residual. This is what "multi" in multi-ets refers to — not multivariate series, not multiple seasonal periods.
_Avoid_: Multivariate loss, multi-horizon residual

**Classical exponential smoothing**:
The traditional Holt/Holt-Winters formulation: a deterministic level/trend/season recursion whose smoothing parameters are fit by minimizing sum-of-squared one-step-ahead residuals (SSE). `multi_ets` and statsmodels' `ExponentialSmoothing` are both classical in this sense; they differ only in *which* residuals the loss is computed over (one-step vs joint multi-step).
_Avoid_: Plain ETS, simple exponential smoothing (that name is reserved for the no-trend-no-season case specifically)

**Innovations ETS**:
The state-space formulation (Hyndman et al.) that models an explicit error component (additive or multiplicative) and is fit by maximum likelihood, e.g. statsmodels' `ETSModel` and statsforecast's `AutoETS`. Structurally different from classical exponential smoothing, not just a different optimizer — comparisons against it are informative but not apples-to-apples with `multi_ets`.
_Avoid_: ETS (ambiguous — always qualify as "classical" or "innovations")

**Matched-structure comparison**:
A benchmark methodology where every model under comparison is forced into the same trend/seasonal configuration (same seasonal period, same additive/multiplicative choice, same damping), isolating the effect of the fitting method (loss horizon, optimizer) from the effect of model selection.
_Avoid_: Fair comparison, apples-to-apples (too vague on their own — say what's held constant)

**Auto-selection comparison**:
A benchmark methodology where each library picks its own model structure via its native selection procedure (e.g. statsforecast's `AutoETS`, `multi_ets`'s `select_model`), reflecting how a user would run the library unassisted. Kept separate from the matched-structure comparison because it conflates fitting-method and model-selection effects.
