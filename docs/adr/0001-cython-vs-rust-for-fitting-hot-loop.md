# 0001: Cython vs. Rust for the fitting hot loop

## Status

Accepted (investigation only — no code from this spike is shipped; see #21).

## Context

#18/#19 replaced JAX (autodiff + `jit` + `jaxopt`'s vmapped multistart) with a
plain-NumPy hand-derived-gradient implementation, and #20 removed the `jax`/
`jaxopt` dependency entirely. That NumPy version is now the only fitting path
`multi_ets` has. It no longer pays JAX's dispatch/compile overhead, but the
recursion (`_states_np`), the BPTT gradient (`_states_bptt_grad_np`), and the
multistart loop over `scipy.optimize.minimize` (`Holt.fit()`) still run as
interpreted Python, including a Python `for` loop over the training series'
length inside every objective/gradient evaluation.

This ticket asks whether compiling that hot loop — via Cython or Rust/PyO3 —
is worth the added packaging/build complexity, on the two shapes this
investigation was scoped to characterize:

1. **Many tiny fits**: lots of small refits in a loop (e.g. a rolling-origin
   backtest) — dispatch/interpreter-overhead-dominated.
2. **One long fit**: a single fit on a longer history with a longer horizon —
   compute-dominated.

## What was benchmarked

Both prototypes port only the **objective+gradient** function — the
undamped-Holt recursion + hand-derived BPTT gradient for `loss="mse"`
(`_holt_forward_np` + `_states_bptt_grad_np` + `_holt_objective_and_grad` in
`src/multi_ets/holt.py`, specialized to `kind="undamped"`, phi fixed at 1.0,
one loss). The **outer multistart loop stays in Python** in both prototypes —
each restart still calls Python's `scipy.optimize.minimize(..., jac=True)`,
which itself calls back into the compiled `objective_and_grad` once per
L-BFGS-B iteration. This is the realistic comparison: rewriting `scipy`'s own
optimizer driver in Rust/Cython too is a much bigger undertaking than this
spike's budget, and it's not where the interpreter overhead the ticket is
asking about actually lives — the per-iteration recursion/gradient loop is.

Both prototypes were sanity-checked against the existing NumPy
`_holt_objective_and_grad` on the same `(params, y, h)` input before trusting
their timings: loss and gradient agree to `atol=1e-10` for both.

Full prototype code, build steps, and the benchmark script live in
`spikes/issue-21-cython-vs-rust/` (see that directory's `README.md`).

### Benchmark parameters

The ticket names "~1000 fits" for the many-tiny shape and "n=700, h=45" for
the one-long shape. The NumPy baseline turned out to cost ~116 ms/fit at the
many-tiny shape's size and ~3.7 s for the one long fit — running the ticket's
literal 1000 reps × 5 trials would have taken over an hour for the NumPy arm
alone. Scaled down for spike turnaround (per-fit/per-iteration overhead
ratios between implementations don't depend on how many times the loop
outer-repeats, so this doesn't change what's being measured):

- Many-tiny: 200 fits/trial (not 1000), 3 trials, `n=40`, `h=3`, `n_starts=5`.
- One-long: 5 trials, `n=700`, `h=45`, `n_starts=10` (ticket's exact shape).

### Results

**Many-tiny** (200 fits, n=40, h=3, n_starts=5 — median of 3 trials, plus the first/cold-start trial):

| impl    | first  | median  | ms/fit |
|---------|-------:|--------:|-------:|
| numpy   | 38.0 s | 23.2 s  | 115.8  |
| cython  |  2.6 s |  2.6 s  |  12.7  |
| rust    |  3.0 s |  2.6 s  |  12.9  |

**One-long** (n=700, h=45, n_starts=10 — median of 5 trials):

| impl    | first   | median  |
|---------|--------:|--------:|
| numpy   | 3.73 s  | 3.68 s  |
| cython  | 0.12 s  | 0.13 s  |
| rust    | 0.23 s  | 0.26 s  |

### Reading the numbers

- Both compiled prototypes deliver a large win on both shapes: ~9x on
  many-tiny, ~28x (Cython) / ~14x (Rust) on one-long.
- Cython and Rust are close on many-tiny (12.7 vs 12.9 ms/fit) — at that
  series length, `scipy.optimize.minimize`'s own per-iteration Python-level
  overhead (setting up each L-BFGS-B call, checking convergence, marshalling
  the callback) dominates enough that it doesn't matter much which compiled
  backend answers the callback.
- Cython noticeably beats Rust on one-long (0.13 s vs 0.26 s, ~2x). The likely
  cause: the PyO3 boundary in this prototype converts the input `y`/`params`
  to/from Python `list`s (`Vec<f64>`) on every call, an O(n) copy that scales
  with series length, while the Cython version takes a NumPy array argument
  directly via a zero-copy typed buffer. This is a prototype-specific
  inefficiency, not a fundamental Rust/PyO3 limitation — a version using
  `numpy`'s `PyArray`/`PyReadonlyArray1` bindings instead of `Vec<f64>` would
  likely close most of this gap. Not fixed here since it doesn't change the
  recommendation below.
- The "first" (cold-start) numbers matter for the many-tiny shape's realistic
  use case (a rolling-origin backtest starting a fresh process): Cython's
  import is a one-time `.so` load (~2.6 s here, in the same ballpark as its
  own steady-state trial — no separate JIT warmup, unlike JAX's old behavior).
  Rust's `first` (3.0 s) is similar. Neither has a meaningfully worse cold
  start than its own steady state, unlike the JAX regime #15/#17 dealt with.

## Decision

**Cython, if/when a follow-on ticket decides the speedup is worth shipping —
skip Rust.**

Reasoning:
- The speedup is real and large enough (9-28x) to matter for the many-tiny
  workload (rolling-origin backtests, hyperparameter search over `n_starts`,
  etc.) — this isn't a "NumPy is already good enough, don't bother" outcome.
- Cython matches or beats Rust on both shapes in this spike, and its
  packaging story is simpler for this codebase specifically: it builds
  against the same NumPy arrays already flowing through `holt.py` with a
  zero-copy typed-buffer interface, needs no additional Python-side
  marshalling layer, and its build step is a NumPy/setuptools primitive the
  project already has a `pyproject.toml` for (Rust needs `maturin` and a
  system Rust toolchain as a wheel-build-time dependency — more moving parts
  for CI/release wheels, for no measured benefit here).
- Rust's disadvantage in this spike (the list-based PyO3 boundary) is fixable,
  but fixing it just to match what Cython already does for free isn't a good
  use of engineering time when Cython already wins.

**Explicitly not decided here**: whether to actually ship this. Per the
ticket, that's a follow-on decision — this spike only establishes that
Cython is the better of the two compiled options *if* the project decides
compiling the hot loop is worth doing. Shipping would mean: porting the
seasonal recursion too (not just undamped-Holt/mse), adding a C-toolchain
build step to CI/release wheels, and deciding whether `multi_ets` is willing
to trade "installs as a pure-Python wheel" for the speedup — a real cost this
spike didn't have to weigh since neither prototype is being shipped.

## Consequences

- No production code changes. `spikes/issue-21-cython-vs-rust/` remains an
  unshipped, unwired reference implementation for whoever picks up the
  follow-on "should we actually ship this" decision.
- If that follow-on happens, start from the Cython prototype here rather than
  from scratch, and extend it to the seasonal recursion and remaining losses
  before assuming the same 9-28x holds across the board.
