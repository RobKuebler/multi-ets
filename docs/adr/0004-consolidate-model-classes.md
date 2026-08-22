# 0004: Consolidate SES/Holt/DampedHolt/HoltWinters/DampedHoltWinters into one class

## Status

Accepted.

## Context

`multi_ets` shipped five public classes -- `SES`, `Holt`, `DampedHolt`,
`HoltWinters`, `DampedHoltWinters` -- one per (trend, seasonal) combination.
Internally they already shared almost everything: a single `_kind` string
picked which row of the `_UNPACK_BY_KIND`/`_PACK_GRAD_BY_KIND` (and seasonal
equivalent) dispatch tables to use, and `SES` itself was already implemented
as `Holt` with `beta`/`phi`/`trend0` pinned to 0 via `_UNPACK_SES` -- the
same parameter-pinning trick the class hierarchy existed to hide. The
`fit()`/`predict()` bodies for the trend-bearing classes differed only in
which bounds/heuristic-init/predict-formula branch they took, not in any
class-specific logic that didn't already dispatch on `_kind`.

That five-classes-one-mechanism shape had a real cost: adding the
no-trend-seasonal kind (issue #38, a level+seasonal model with no trend term)
had no home in the hierarchy -- it wasn't a `HoltWinters` (which always
carries a trend) and wasn't any existing subclass. It also meant every
future (trend, seasonal) combination would need its own class, each
re-deriving bounds/init/predict from the same underlying dispatch tables the
existing classes already used internally.

The alternative considered was keeping the five classes and giving the new
no-trend-seasonal kind a sixth (`SeasonalNoTrend` or similar). That preserves
`isinstance(model, HoltWinters)`-style checks and keeps each class's
constructor signature minimal (no unused `seasonal_period`/`seasonal_type`
params on `SES`/`Holt`/`DampedHolt`). But it locks in one class per grid
cell forever, and statsmodels' own `ExponentialSmoothing` / statsforecast's
`AutoETS` both use the single-class-plus-parameters convention instead --
consolidating brings `multi_ets` in line with that convention rather than
against it.

## Decision

**One class.** `ExponentialSmoothing(h, trend=None|"add"|"damped",
seasonal_period=None, seasonal_type="additive", loss=..., n_starts=...,
...)` replaces all five. `trend` and `seasonal_period` together resolve to
the same internal `_kind` string the old classes computed via their class
attributes; `fit()`/`predict()`/`predict_interval()` branch on
`_kind in _SEASONAL_KINDS` once, the same way `_resid_std()` already did
before this change.

The five old classes are removed outright, with no deprecated aliases:
`multi_ets` is pre-1.0 with no evidence of external consumers, so there's no
backwards-compatibility surface to preserve, and a compatibility shim would
just be more code carrying the exact class-per-configuration structure this
decision is undoing.

This traded discoverability for consistency:

- **Lost**: `isinstance(model, HoltWinters)` no longer works -- callers
  needing to know "does this candidate have a trend/season" check
  `model.trend`/`model.seasonal_period` instead (see `select_model`'s own
  tests for the migration). A constructor typo like `seasonal_period=12` on
  what was meant to be a non-seasonal model no longer fails at the type
  level (wrong class) -- it's caught at `fit()`/validation time instead, if
  at all.
- **Gained**: one discoverable constructor instead of five class names to
  remember; a new (trend, seasonal) combination is a new `_kind` string plus
  its dispatch-table entries, not a new class; and the no-trend-seasonal
  kind (#38) has an obvious home -- `trend=None` with `seasonal_period` set
  -- instead of needing its own class or bolting onto an existing one that
  doesn't fit.

## Consequences

- `AutoExponentialSmoothing` (#40) builds its candidate grid by varying
  `trend`/`seasonal_type` on one class, not by importing five class names --
  a wrapper around `select_model` would have been much more awkward against
  the old hierarchy.
- Any future (trend, seasonal, ...) axis extension is a new `_kind` plus
  dispatch-table entries and a widened `trend`/`seasonal_type` validation,
  not a new class.
- If `multi_ets` later wants isinstance-style discoverability back (e.g. for
  a plugin system matching on model family), that's a reason to revisit this
  ADR specifically -- it would mean re-introducing some structural split,
  weighed against the cost this decision removed.
