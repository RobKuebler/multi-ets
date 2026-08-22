"""Verifies the _holt_hot Cython core (issue #22) against holt.py's NumPy path.

Skips cleanly if the extension hasn't been built (uv pip install cython &&
uv run python setup.py build_ext --inplace in src/multi_ets/, see setup.py).
"""

import numpy as np
import pytest

from multi_ets import holt as holt_module

_holt_hot = pytest.importorskip("multi_ets._holt_hot")

_LOSS_ID = {"mse": 0, "mae": 1, "huber": 2, "pinball": 3}
_KINDS = ("undamped", "damped", "ses")
_LOSSES = ("mse", "mae", "huber", "pinball")
_SEED = {(loss, kind): i for i, (loss, kind) in enumerate((L, K) for L in _LOSSES for K in _KINDS)}
_DELTA = 0.7  # deliberately not the default 1.0, to exercise huber's clip boundary
_TAU = 0.3  # deliberately not the default 0.5, to exercise pinball's asymmetry


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
    loss_key = holt_module._loss_key(loss, _DELTA, _TAU)

    value_np, grad_np = holt_module._holt_objective_and_grad(params, y, h, kind, loss_key)

    full5 = np.asarray(holt_module._UNPACK_BY_KIND[kind](params), dtype=float)
    value_c, grad5_c = _holt_hot.objective_and_grad(full5, y, h, _LOSS_ID[loss], _DELTA, _TAU)
    grad_c = holt_module._PACK_GRAD_BY_KIND[kind](grad5_c)

    assert value_c == pytest.approx(value_np, abs=1e-9)
    np.testing.assert_allclose(grad_c, grad_np, atol=1e-9)
