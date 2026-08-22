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
their timings: loss and gradient agree to `atol=1e-10` for both, and a full
multistart fit converges to bit-identical fitted params across all three
implementations.

Full prototype code, build steps, and the benchmark script live in
`spikes/issue-21-cython-vs-rust/` (see that directory's `README.md`).

### Benchmark parameters

The ticket names "~1000 fits" for the many-tiny shape and "n=700, h=45" for
the one-long shape. The NumPy baseline turned out to cost ~100-116 ms/fit at
the many-tiny shape's size and ~3.5-3.7 s for the one long fit — running the
ticket's literal 1000 reps at multiple trials would have taken over an hour
for the NumPy arm alone. Scaled down for spike turnaround (per-fit/per-
iteration overhead ratios between implementations don't depend on how many
times the loop outer-repeats, so this doesn't change what's being measured):

- Many-tiny: 200 fits/trial (not 1000), 5 trials, `n=40`, `h=3`, `n_starts=5`.
- One-long: 15 trials, `n=700`, `h=45`, `n_starts=10` (ticket's exact shape).

### Results, round 1: naive PyO3 binding

**Many-tiny** (200 fits, n=40, h=3, n_starts=5):

| impl    | first  | steady state | ms/fit |
|---------|-------:|-------------:|-------:|
| numpy   | 38.0 s | 23.2 s       | 115.8  |
| cython  |  2.6 s |  2.6 s       |  12.7  |
| rust    |  3.0 s |  2.6 s       |  12.9  |

**One-long** (n=700, h=45, n_starts=10):

| impl    | first   | steady state |
|---------|--------:|-------------:|
| numpy   | 3.73 s  | 3.68 s       |
| cython  | 0.12 s  | 0.13 s       |
| rust    | 0.23 s  | 0.26 s       |

Both compiled prototypes already won big here: ~9x on many-tiny, ~28x
(Cython) / ~14x (Rust) on one-long. But Cython noticeably beat Rust on
one-long (0.13 s vs 0.26 s, ~2x) — worth understanding before recommending
one over the other on a performance basis.

### Follow-up: is Rust's gap fundamental, or a naive-binding artifact?

The round-1 Rust prototype took `y: Vec<f64>`, which forces PyO3 to copy the
entire series out of the NumPy array into a fresh Rust `Vec` on *every call*
— and `objective_and_grad` is called once per L-BFGS-B iteration, so a
700-element series gets re-copied hundreds of times per fit. The Cython
prototype never paid this: `double[:]` typed memoryviews give zero-copy
direct access to the NumPy buffer. This looks like "Rust is slower" but is
actually "this particular binding forgot to ask for zero-copy access," so it
was worth fixing before trusting the round-1 verdict. Four changes, each
re-verified to still match the NumPy baseline exactly (`atol=1e-10` on
objective+gradient; a full multistart fit converges to bit-identical params
vs. NumPy, checked on 5 more random series):

1. **Zero-copy array access**: `y: Vec<f64>` → `y: numpy::PyReadonlyArray1<f64>`
   (the `numpy` crate), read via `.as_slice()` — no copy, mirrors Cython's
   memoryview exactly.
2. **LTO + `codegen-units = 1`** in `Cargo.toml`'s release profile — standard
   levers for small hot-loop crates that `cargo`'s defaults leave off.
3. **Flattened `d_forecast` from `Vec<Vec<f64>>` to a single flat
   `Vec<f64>`** indexed by `t * h + k`. This mattered more than expected: the
   nested-`Vec` version was ~656 separate heap allocations (one per origin
   row) for the one-long shape, dwarfing everything else. Cython's NumPy
   array was always one contiguous allocation; the round-1 Rust code hadn't
   matched that.
4. **`get_unchecked`/`get_unchecked_mut` in the hot loops** (wrapped in
   `unsafe`, with a comment establishing why every index is provably in
   range), matching the `.pyx`'s `boundscheck=False` directive — Rust's safe
   indexing bounds-checks by default; Cython's didn't, once told not to.

### Results, round 2: after the above four changes

Reported as `min` across ~20 repeats, not median: this machine is a shared
desktop (browser, other processes) rather than a dedicated benchmark box —
`uptime` showed a load average of 6+ on 8 cores while these ran, and
wall-clock *median* across a handful of reps swung 2-5x between runs purely
from contention. `min` is far more stable and is what `bench.py` reports.
Take these as "same order of magnitude, Rust ≈ Cython," not as precise to
the decimal.

| impl    | many-tiny ms/fit | one-long min |
|---------|-----------------:|-------------:|
| cython  | 12.1             | 0.115 s      |
| rust    | 13.8             | 0.122 s      |

Rust and Cython are now, for practical purposes, tied on both shapes. **The
round-1 "Cython beats Rust" result was a binding artifact, not a language
difference** — once the Rust side does what the Cython side was already
doing (zero-copy access, one contiguous buffer, no redundant bounds checks),
the two perform the same.

### Reference: statsmodels / statsforecast wall-clock (not apples-to-apples)

For context, the same two shapes fit via `statsmodels.tsa.holtwinters.Holt`
and `statsforecast.models.Holt` (both single-shot classical 1-step-SSE fits,
no multistart — see CONTEXT.md's "Matched-structure comparison" and
"Classical exponential smoothing" entries for why these aren't fit the same
way multi_ets is):

| impl          | many-tiny ms/fit | one-long min |
|---------------|-----------------:|-------------:|
| statsmodels   | 21.4             | 0.033 s      |
| statsforecast | 1.6              | 0.015 s      |

Both are faster than even the compiled multi_ets prototypes — unsurprising,
since they solve a single convex-ish 1-step SSE problem once rather than a
10-restart multistart search over a non-convex joint multi-step loss. This
is a speed reference only, not a signal that multi_ets's approach is
inefficient for what it's actually solving; it's solving a harder problem
(joint multi-step loss, multiple random restarts) by design (see
CONTEXT.md's "Multi-step joint loss" entry), and that's orthogonal to
whether its implementation is compiled or interpreted.

## Decision

**A compiled hot loop is clearly worth it if this is ever shipped — and,
once both bindings are written properly, Cython and Rust perform the same.
Recommend Cython anyway, on packaging grounds, not performance.**

Reasoning:
- The speedup over the NumPy baseline is real and large enough (9-28x) to
  matter for the many-tiny workload (rolling-origin backtests,
  hyperparameter search over `n_starts`, etc.) — this isn't a "NumPy is
  already good enough, don't bother" outcome.
- Once written to avoid the same classes of overhead (zero-copy array
  access, one contiguous buffer, no redundant bounds checks), Cython and
  Rust are performance-equivalent for this hot loop. Neither language has an
  inherent edge here — the round-1 gap was this spike's Rust binding being
  naive, not Rust being slow.
- With performance a tie, the deciding factor is packaging/toolchain cost:
  Cython builds against the same NumPy arrays already flowing through
  `holt.py` via a zero-copy typed-buffer interface, needs no extra
  Python-side marshalling layer, and its build step (`setuptools` +
  `Cython.Build.cythonize`) is a primitive the Python packaging ecosystem
  (and this project's `pyproject.toml`) already assumes. Rust needs
  `maturin` and a system Rust toolchain as a wheel-build-time dependency —
  more moving parts for CI/release wheels, for no performance benefit once
  Rust is done properly.

**Explicitly not decided here**: whether to actually ship this. Per the
ticket, that's a follow-on decision — this spike only establishes that (a) a
compiled hot loop is worth it if the project wants the speedup, and (b)
between Cython and Rust, pick on packaging grounds since performance won't
decide it. Shipping would mean: porting the seasonal recursion too (not just
undamped-Holt/mse), adding a C-toolchain build step to CI/release wheels,
and deciding whether `multi_ets` is willing to trade "installs as a
pure-Python wheel" for the speedup — a real cost this spike didn't have to
weigh since neither prototype is being shipped.

## Consequences

- No production code changes. `spikes/issue-21-cython-vs-rust/` remains an
  unshipped, unwired reference implementation for whoever picks up the
  follow-on "should we actually ship this" decision.
- If that follow-on happens, start from the Cython prototype here rather
  than from scratch (per the packaging-cost reasoning above), and extend it
  to the seasonal recursion and remaining losses before assuming the same
  9-28x holds across the board. If Rust is preferred for other reasons
  (team familiarity, an existing Rust toolchain in CI, etc.), the round-2
  `rust_proto/src/lib.rs` is a fine starting point instead — just carry
  forward its zero-copy/flat-buffer/unsafe-indexing choices rather than
  reverting to the round-1 approach.
