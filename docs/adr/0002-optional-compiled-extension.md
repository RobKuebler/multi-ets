# 0002: Compiled hot loop ships as an optional `[fast]` extra

## Status

Accepted.

## Context

ADR 0001 established that a compiled hot loop is worth shipping and picked
Cython over Rust on packaging grounds, but explicitly left one question for
a follow-on decision: does `multi_ets` ship the compiled extension as a
**required** build dependency (the package stops installing as a
pure-Python wheel), or as an **optional** extra (NumPy-only by default, the
compiled backend opt-in)?

Issues #22/#23 built the actual Cython core (`src/multi_ets/_holt_hot.pyx`,
`objective_and_grad`/`seasonal_objective_and_grad`) covering every (kind,
loss) combination the NumPy path supports. Issue #24 wires it into
`fit()` and needs this question answered to know how to build/ship it.

## Decision

**Optional extra.** `multi-ets` stays a pure-Python wheel by default,
installable and fully correct with zero C toolchain via the existing NumPy
path in `holt.py`. The compiled backend is available via
`pip install multi-ets[fast]` / `uv sync --extra fast`, which pulls in
`cython` as a build tool.

`Holt.fit()`/`SES.fit()`/`DampedHolt.fit()`/`HoltWinters.fit()`/
`DampedHoltWinters.fit()` each dispatch to the compiled
`multi_ets._holt_hot.objective_and_grad`/`seasonal_objective_and_grad` when
(a) the extension is importable and (b) `loss` resolves to a built-in key
(not a custom callable); otherwise they use the existing NumPy closure. The
import is guarded once at module load (`try: from multi_ets import
_holt_hot / except ImportError`), not per-call, so a missing extension never
raises -- it silently falls back.

### Packaging reality: `uv_build` cannot compile `_holt_hot.pyx`

This project's build backend is `uv_build` (`pyproject.toml`'s
`[build-system]`). `uv_build` is deliberately a pure-Python-only build
backend -- it has no build-hook mechanism for running `Cython.Build.cythonize`
or a C compiler as part of `pip install` / `uv sync`. Wiring `[fast]` as an
`[project.optional-dependencies]` entry (done: adds `cython` to the
environment) therefore does **not**, by itself, produce a built
`_holt_hot` extension -- `cython` merely becomes available on `PATH`/as an
importable package. The extension still has to be compiled by a separate,
manual step:

```bash
uv sync --extra fast          # or: pip install -e ".[fast]"
cd src/multi_ets && uv run python setup.py build_ext --inplace
```

This is the "documented compile from source" path the ticket names as the
acceptable minimum for an optional extra (as opposed to full multi-platform
`cibuildwheel` wheel building, which is proportionate to a *required*
compiled dependency, not an opt-in one). Switching build backends (e.g. to
`scikit-build-core`, `maturin`, or plain `setuptools` with a
`cythonize`-driven `build_ext` hook) would let `[fast]` actually trigger the
compile step during install -- that's a real option if this decision is
revisited, but out of scope here per "lightest correct thing" for an
optional extra.

### CI

The existing `.github/workflows/ci.yml` gained a `test-fast` job: builds the
`[fast]` extra (`uv sync --extra fast` + the manual `build_ext --inplace`
step above) and runs `tests/test_holt_hot.py`/`tests/test_seasonal_hot.py`
against the compiled path, so the compiled backend doesn't silently bit-rot.
This is the minimum CI investment proportionate to an optional extra; it
does not attempt cross-platform wheel building.

## Consequences

- No C toolchain is required to install or run `multi_ets`; the compiled
  path is purely an opt-in speedup.
- Anyone wanting the speedup must run the manual `build_ext --inplace` step
  themselves (README documents this) -- `pip install multi-ets[fast]` alone
  is not sufficient, and that's a real, documented limitation of building on
  `uv_build` rather than a papered-over gap.
- If `multi_ets` ever needs the compiled backend to be the *default* (not
  opt-in), revisit both this ADR and the build-backend choice together --
  `uv_build`'s pure-Python-only design is the blocker, not anything specific
  to Cython.
