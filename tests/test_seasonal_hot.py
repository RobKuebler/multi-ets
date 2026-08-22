"""Verifies the _holt_hot seasonal core (issue #23) against holt.py's NumPy path.

Skips cleanly if the extension hasn't been built (uv pip install cython &&
uv run python setup.py build_ext --inplace in src/multi_ets/, see setup.py).
"""

import numpy as np
import pytest

from multi_ets import holt as holt_module

_holt_hot = pytest.importorskip("multi_ets._holt_hot")

_LOSS_ID = {"mse": 0, "mae": 1, "huber": 2, "pinball": 3}
_KINDS = ("seasonal", "damped_seasonal", "seasonal_mult", "damped_seasonal_mult")
_LOSSES = ("mse", "mae", "huber", "pinball")
_SEED = {(loss, kind): i for i, (loss, kind) in enumerate((L, K) for L in _LOSSES for K in _KINDS)}
_DELTA = 0.7  # deliberately not the default 1.0, to exercise huber's clip boundary
_TAU = 0.3  # deliberately not the default 0.5, to exercise pinball's asymmetry
_M = 4  # season length


def _random_params(kind: str, rng: np.random.Generator, multiplicative: bool) -> np.ndarray:
    alpha = rng.uniform(0.05, 0.95)
    beta = rng.uniform(0.05, 0.95)
    gamma = rng.uniform(0.05, 0.95)
    # Multiplicative divides by level/season, so keep both comfortably positive
    # (this only needs to avoid degenerate near-zero divisors, not be a
    # realistic fit -- both paths run the identical arithmetic either way).
    level0 = rng.uniform(15, 25) if multiplicative else rng.normal(0, 5)
    trend0 = rng.normal(0, 0.2) if multiplicative else rng.normal(0, 1)
    season0 = rng.uniform(0.7, 1.3, _M) if multiplicative else rng.normal(0, 1, _M)
    if kind in ("seasonal", "seasonal_mult"):
        return np.concatenate([[alpha, beta, gamma, level0, trend0], season0])
    phi = rng.uniform(0.05, 0.9)
    return np.concatenate([[alpha, beta, phi, gamma, level0, trend0], season0])


@pytest.mark.parametrize("kind", _KINDS)
@pytest.mark.parametrize("loss", _LOSSES)
def test_matches_numpy_reference(loss: str, kind: str) -> None:
    rng = np.random.default_rng(_SEED[loss, kind])
    multiplicative = kind in holt_module._MULTIPLICATIVE_KINDS
    y = rng.uniform(15, 25, 40) if multiplicative else np.cumsum(rng.normal(0.5, 1.0, 40)) + 20
    h = 5
    params = _random_params(kind, rng, multiplicative)
    loss_key = holt_module._loss_key(loss, _DELTA, _TAU)

    value_np, grad_np = holt_module._seasonal_objective_and_grad(params, y, h, kind, _M, loss_key)

    alpha, beta, phi, gamma, level0, trend0, season0 = holt_module._UNPACK_SEASONAL_BY_KIND[kind](
        params, _M
    )
    full = np.concatenate([[alpha, beta, phi, gamma, level0, trend0], season0])
    value_c, grad_c_full = _holt_hot.seasonal_objective_and_grad(
        full, y, h, _M, int(multiplicative), _LOSS_ID[loss], _DELTA, _TAU
    )
    grad_c = holt_module._PACK_GRAD_SEASONAL_BY_KIND[kind](
        (
            grad_c_full[0],
            grad_c_full[1],
            grad_c_full[2],
            grad_c_full[3],
            grad_c_full[4],
            grad_c_full[5],
            grad_c_full[6:],
        )
    )

    assert value_c == pytest.approx(value_np, abs=1e-12)
    np.testing.assert_allclose(grad_c, grad_np, atol=1e-12)
