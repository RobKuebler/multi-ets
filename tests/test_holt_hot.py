"""Verifies the _holt_hot Cython core (issue #22) against holt.py's NumPy path.

Skips cleanly if the extension hasn't been built (uv pip install cython &&
uv run python setup.py build_ext --inplace in src/multi_ets/, see setup.py).
"""

import numpy as np
import pytest

from multi_ets import ExponentialSmoothing
from multi_ets import holt as holt_module

_holt_hot = pytest.importorskip("multi_ets._holt_hot")

_LOSS_ID = holt_module._LOSS_ID
_KINDS = ("undamped", "damped", "ses")
_LOSSES = (
    "mse",
    "mae",
    "huber",
    "pinball",
    "poisson",
    "negbinom",
    "tweedie",
    "expectile",
    "logcosh",
)
_SEED = {(loss, kind): i for i, (loss, kind) in enumerate((L, K) for L in _LOSSES for K in _KINDS)}
_DELTA = 0.7  # deliberately not the default 1.0, to exercise huber's clip boundary
_TAU = 0.3  # deliberately not the default 0.5, to exercise pinball's/expectile's asymmetry
_NB_ALPHA = 0.8  # deliberately not the default 1.0, to exercise negbinom's dispersion term
_TW_P = 1.3  # deliberately not the default 1.5, to exercise tweedie's power parameter


def _random_params(kind: str, rng: np.random.Generator) -> np.ndarray:
    alpha = rng.uniform(0.05, 0.95)
    level0 = rng.normal(0, 5)
    trend0 = rng.normal(0, 1)
    if kind == "ses":
        return np.array([alpha, level0])
    beta = rng.uniform(0.05, 0.95)
    if kind == "undamped":
        return np.array([alpha, beta, level0, trend0])
    phi = rng.uniform(0.05, 0.9)
    return np.array([alpha, beta, phi, level0, trend0])


@pytest.mark.parametrize("kind", _KINDS)
@pytest.mark.parametrize("loss", _LOSSES)
def test_matches_numpy_reference(loss: str, kind: str) -> None:
    rng = np.random.default_rng(_SEED[loss, kind])
    y = np.cumsum(rng.normal(0.5, 1.0, 40)) + 20
    h = 5
    params = _random_params(kind, rng)
    loss_key = holt_module._loss_key(loss, _DELTA, _TAU, _NB_ALPHA, _TW_P)

    value_np, grad_np = holt_module._holt_objective_and_grad(params, y, h, kind, loss_key)

    full5 = np.asarray(holt_module._UNPACK_BY_KIND[kind](params), dtype=float)
    value_c, grad5_c = _holt_hot.objective_and_grad(
        full5, y, h, _LOSS_ID[loss], _DELTA, _TAU, _NB_ALPHA, _TW_P
    )
    grad_c = holt_module._PACK_GRAD_BY_KIND[kind](grad5_c)

    assert value_c == pytest.approx(value_np, abs=1e-9)
    np.testing.assert_allclose(grad_c, grad_np, atol=1e-9)


@pytest.mark.parametrize("loss", _LOSSES)
@pytest.mark.parametrize("trend", ["add", None, "damped"])
def test_fit_dispatches_to_compiled_backend(
    trend: str | None, loss: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """fit() with a built-in loss must actually call the compiled objective_and_grad.

    A numerics-only test wouldn't catch "the dispatch silently never
    triggers" (fit() would still converge correctly via the NumPy path) --
    this spies on the compiled entry point to prove it's reached. Parametrized
    over every built-in loss (#32): the #26-30 losses need this exercised
    end-to-end too, not just their objective_and_grad numerics above.
    """
    rng = np.random.default_rng(0)
    y = np.cumsum(rng.normal(0.5, 1.0, 40)) + 20
    calls = []
    real = _holt_hot.objective_and_grad

    def spy(*args: object, **kwargs: object) -> tuple[float, np.ndarray]:
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(_holt_hot, "objective_and_grad", spy)
    ExponentialSmoothing(h=3, trend=trend, n_starts=2, loss=loss).fit(y)
    assert calls, "fit() did not call the compiled backend"


def test_fit_with_custom_loss_does_not_dispatch(monkeypatch: pytest.MonkeyPatch) -> None:
    """A custom callable loss must stay on the NumPy path, never the compiled one."""
    rng = np.random.default_rng(0)
    y = np.cumsum(rng.normal(0.5, 1.0, 40)) + 20
    calls = []
    monkeypatch.setattr(
        _holt_hot, "objective_and_grad", lambda *a, **k: calls.append(1) or (0.0, np.zeros(5))
    )
    ExponentialSmoothing(
        h=3, trend="add", n_starts=2, loss=lambda yt, yp: ((yt - yp) ** 2).mean()
    ).fit(y)
    assert not calls, "custom loss must not reach the compiled backend"


@pytest.mark.parametrize("kind", _KINDS)
@pytest.mark.parametrize("loss", _LOSSES)
def test_matches_numpy_reference_with_nan_actuals(loss: str, kind: str) -> None:
    """#35: the compiled core's skip-and-carry + NaN-actual masking must match the NumPy path."""
    rng = np.random.default_rng(_SEED[loss, kind] + 100)
    y = np.cumsum(rng.normal(0.5, 1.0, 40)) + 20
    y[5] = np.nan
    y[20] = np.nan
    y[21] = np.nan
    h = 5
    params = _random_params(kind, rng)
    loss_key = holt_module._loss_key(loss, _DELTA, _TAU, _NB_ALPHA, _TW_P)

    value_np, grad_np = holt_module._holt_objective_and_grad(params, y, h, kind, loss_key)

    full5 = np.asarray(holt_module._UNPACK_BY_KIND[kind](params), dtype=float)
    value_c, grad5_c = _holt_hot.objective_and_grad(
        full5, y, h, _LOSS_ID[loss], _DELTA, _TAU, _NB_ALPHA, _TW_P
    )
    grad_c = holt_module._PACK_GRAD_BY_KIND[kind](grad5_c)

    assert value_c == pytest.approx(value_np, abs=1e-9)
    np.testing.assert_allclose(grad_c, grad_np, atol=1e-9)


@pytest.mark.parametrize("trend", ["add", None, "damped"])
def test_fit_dispatches_to_compiled_backend_for_nan_containing_y(
    trend: str | None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """#35: a NaN-containing y must still use the compiled backend, not fall back to NumPy."""
    rng = np.random.default_rng(0)
    y = np.cumsum(rng.normal(0.5, 1.0, 40)) + 20
    y[10] = np.nan
    calls = []
    real = _holt_hot.objective_and_grad

    def spy(*args: object, **kwargs: object) -> tuple[float, np.ndarray]:
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(_holt_hot, "objective_and_grad", spy)
    ExponentialSmoothing(h=3, trend=trend, n_starts=2, loss="mse").fit(y)
    assert calls, "fit() did not call the compiled backend for a NaN-containing y"
