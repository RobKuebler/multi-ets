# Issue #21 spike: Cython vs. Rust for the fitting hot loop

Not shipped code — a throwaway investigation. See `docs/adr/0001-cython-vs-rust-for-fitting-hot-loop.md`
for the numbers and recommendation. Nothing here is wired into `pyproject.toml`'s
build or imported by `src/multi_ets/`.

What was ported: the undamped-Holt, `loss="mse"` objective+gradient
(`_holt_forward_np` + `_states_bptt_grad_np` + `_holt_objective_and_grad` in
`src/multi_ets/holt.py`), specialized to `kind="undamped"` and one loss. The
outer multistart loop (`for start in _start_grid(...): scipy.optimize.minimize(...)`)
stays in Python in both prototypes — only the per-iteration objective+gradient
call is compiled. See the ADR for why that's the interesting comparison.

## Layout

- `cython_proto/holt_hot.pyx` — Cython port, built via `setup.py`.
- `rust_proto/` — Rust/PyO3 port (`src/lib.rs`), built via `maturin`.
- `bench.py` — sanity-checks both against the NumPy baseline, then benchmarks
  all three on the two shapes from the ticket.

## Build

From the repo root, with the project's `.venv` active (`uv run ...` picks it up automatically):

```sh
# Cython
cd spikes/issue-21-cython-vs-rust/cython_proto && uv run python setup.py build_ext --inplace

# Rust (needs a Rust toolchain: rustc/cargo)
cd spikes/issue-21-cython-vs-rust/rust_proto
VIRTUAL_ENV=$(pwd)/../../../.venv PATH="$(pwd)/../../../.venv/bin:$PATH" maturin develop --release
```

Don't run plain `uv run maturin develop` from inside `rust_proto/` — that
directory has its own `pyproject.toml` (maturin's build backend config), so
`uv run` treats it as a separate project boundary and creates a throwaway
`.venv` right there instead of using the repo's `.venv`, and the built
extension ends up somewhere `bench.py` can't import it from. Pointing
`VIRTUAL_ENV`/`PATH` at the repo's own `.venv` (as above) makes `maturin`
install straight into it (shows up in `uv pip list` as `holt-hot-rs`). The
Cython build produces `cython_proto/holt_hot*.so` next to the `.pyx`, picked
up via the `sys.path.insert(...)` at the top of `bench.py`.

Both `cython` and `maturin` were installed with `uv pip install cython maturin`
directly into the project `.venv` for this spike — not added to
`pyproject.toml`/`uv.lock`, since neither is a real dependency of anything
shipped.

## Run

```sh
uv run python spikes/issue-21-cython-vs-rust/bench.py
```

Prints a correctness sanity check (each prototype's `objective_and_grad`
matches `_holt_objective_and_grad`'s NumPy output to `atol=1e-10`), then the
benchmark numbers for both shapes × three implementations.
