# 0005: Ship the compiled hot loop by default, as prebuilt wheels

## Status

Accepted. Supersedes [0002](0002-optional-compiled-extension.md).

## Context

ADR 0002 shipped the Cython hot loop (`_holt_hot.pyx`) as an opt-in `[fast]`
extra. Because the build backend was `uv_build`, which only builds
pure-Python packages, that extra just installed Cython: users still had to run
`setup.py build_ext --inplace` by hand. In practice almost nobody would get
the 9-28x speedup (ADR 0001) from `pip install multi-ets`.

The goal is that `pip install multi-ets` gives the compiled version with no
extra steps, while still installing on machines without a C compiler. ADR 0002
named the blocker: the build backend, not Cython.

## Decision

- **Build backend: setuptools + Cython.** `pyproject.toml` keeps all metadata;
  a root `setup.py` only declares the `multi_ets._holt_hot` extension. Every
  install (wheel build, sdist install, `uv sync`, `pip install -e .`)
  compiles it. Build requirements are `setuptools`, `cython` and
  `numpy>=2.0` (extensions built against NumPy 2 also run on NumPy 1.x, so the
  runtime `numpy>=1.26` floor is unchanged).
- **Compiling is best-effort by default.** `setup.py`'s `build_ext` turns a
  compile failure into a warning and installs the pure-Python package, which
  already falls back to the NumPy path when `_holt_hot` can't be imported.
  The `MULTI_ETS_EXTENSION` environment variable switches this: `required`
  (CI and release builds, so a broken compile can't ship silently) or `skip`
  (to test the fallback).
- **Prebuilt wheels on PyPI via cibuildwheel.** `.github/workflows/release.yml`
  builds wheels for every supported CPython on Linux (x86_64, aarch64; glibc
  and musl), macOS (x86_64, arm64) and Windows (x86_64), plus an sdist, and
  publishes them with PyPI trusted publishing when a GitHub release is
  published. Each wheel is checked to contain the extension and runs the
  hot-loop tests. Free-threaded and 32-bit builds are skipped.
- **The `[fast]` extra is removed**; there's nothing left to opt into.
- **Cython, not Rust**, as ADR 0001 recommended. maturin's release tooling is
  comparable, but a source build would need a Rust toolchain, it can't easily
  degrade to pure Python, and the Cython port already covers every loss.

## Consequences

- `pip install multi-ets` on mainstream platforms gets the compiled backend,
  no compiler needed.
- Exotic platforms build from source; without a compiler they get a working
  pure-Python install (with a warning), same as before.
- Releasing is a CI job (build ~a dozen wheels, then publish), not a local
  `uv build && uv publish`. Setting up trusted publishing on PyPI is a
  one-time manual step.
- Development installs need a C compiler unless `MULTI_ETS_EXTENSION=skip`.
- CI runs the full suite twice: with the extension required (Linux, macOS) and
  with it skipped (`test-pure-python`), so neither path rots.
