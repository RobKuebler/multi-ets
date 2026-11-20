"""Verifies the _holt_hot seasonal core (issue #23) against holt.py's NumPy path.

Skips cleanly if the extension isn't built (e.g. MULTI_ETS_EXTENSION=skip, or no
C compiler at install time -- see setup.py).
"""

import numpy as np
import pytest

from multi_ets import ExponentialSmoothing
from multi_ets import holt as holt_module

_holt_hot = pytest.importorskip("multi_ets._holt_hot")

_LOSS_ID = holt_module._LOSS_ID
_KINDS = (
    "seasonal",
    "damped_seasonal",
    "seasonal_no_trend",
    "seasonal_mult",
    "damped_seasonal_mult",
    "seasonal_no_trend_mult",
)
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
    if kind in ("seasonal_no_trend", "seasonal_no_trend_mult"):
        return np.concatenate([[alpha, gamma, level0], season0])
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
    loss_key = holt_module._loss_key(loss, _DELTA, _TAU, _NB_ALPHA, _TW_P)

    value_np, grad_np = holt_module._seasonal_objective_and_grad(params, y, h, kind, _M, loss_key)

    alpha, beta, phi, gamma, level0, trend0, season0 = holt_module._UNPACK_SEASONAL_BY_KIND[kind](
        params, _M
    )
    full = np.concatenate([[alpha, beta, phi, gamma, level0, trend0], season0])
    value_c, grad_c_full = _holt_hot.seasonal_objective_and_grad(
        full, y, h, _M, int(multiplicative), _LOSS_ID[loss], _DELTA, _TAU, _NB_ALPHA, _TW_P
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


@pytest.mark.parametrize("loss", _LOSSES)
@pytest.mark.parametrize("trend", ["add", "damped", None])
@pytest.mark.parametrize("seasonal_type", ["additive", "multiplicative"])
def test_fit_dispatches_to_compiled_backend(
    trend: str | None, seasonal_type: str, loss: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """fit() with a built-in loss must actually call the compiled seasonal_objective_and_grad.

    A numerics-only test wouldn't catch "the dispatch silently never
    triggers" -- this spies on the compiled entry point to prove it's reached.
    Parametrized over every built-in loss (#32): the #26-30 losses need this
    exercised end-to-end too, not just their objective_and_grad numerics above.
    """
    rng = np.random.default_rng(0)
    y = (
        rng.uniform(15, 25, 40)
        if seasonal_type == "multiplicative"
        else np.cumsum(rng.normal(0.5, 1.0, 40)) + 20
    )
    calls = []
    real = _holt_hot.seasonal_objective_and_grad

    def spy(*args: object, **kwargs: object) -> tuple[float, np.ndarray]:
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(_holt_hot, "seasonal_objective_and_grad", spy)
    ExponentialSmoothing(
        h=3, trend=trend, seasonal_period=_M, seasonal_type=seasonal_type, n_starts=2, loss=loss
    ).fit(y)
    assert calls, "fit() did not call the compiled backend"


def test_fit_with_custom_loss_does_not_dispatch(monkeypatch: pytest.MonkeyPatch) -> None:
    """A custom callable loss must stay on the NumPy path, never the compiled one."""
    rng = np.random.default_rng(0)
    y = np.cumsum(rng.normal(0.5, 1.0, 40)) + 20
    calls = []
    monkeypatch.setattr(
        _holt_hot,
        "seasonal_objective_and_grad",
        lambda *a, **k: calls.append(1) or (0.0, np.zeros(5 + _M)),
    )
    ExponentialSmoothing(
        h=3,
        trend="add",
        seasonal_period=_M,
        n_starts=2,
        loss=lambda yt, yp: ((yt - yp) ** 2).mean(),
    ).fit(y)
    assert not calls, "custom loss must not reach the compiled backend"


@pytest.mark.parametrize("kind", _KINDS)
@pytest.mark.parametrize("loss", _LOSSES)
def test_matches_numpy_reference_with_nan_actuals(loss: str, kind: str) -> None:
    """#36: the compiled seasonal core's skip-and-carry + NaN masking must match the NumPy path."""
    rng = np.random.default_rng(_SEED[loss, kind] + 100)
    multiplicative = kind in holt_module._MULTIPLICATIVE_KINDS
    y = rng.uniform(15, 25, 40) if multiplicative else np.cumsum(rng.normal(0.5, 1.0, 40)) + 20
    y[5] = np.nan
    y[20] = np.nan
    y[21] = np.nan
    h = 5
    params = _random_params(kind, rng, multiplicative)
    loss_key = holt_module._loss_key(loss, _DELTA, _TAU, _NB_ALPHA, _TW_P)

    value_np, grad_np = holt_module._seasonal_objective_and_grad(params, y, h, kind, _M, loss_key)

    alpha, beta, phi, gamma, level0, trend0, season0 = holt_module._UNPACK_SEASONAL_BY_KIND[kind](
        params, _M
    )
    full = np.concatenate([[alpha, beta, phi, gamma, level0, trend0], season0])
    value_c, grad_c_full = _holt_hot.seasonal_objective_and_grad(
        full, y, h, _M, int(multiplicative), _LOSS_ID[loss], _DELTA, _TAU, _NB_ALPHA, _TW_P
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

    assert value_c == pytest.approx(value_np, abs=1e-9)
    np.testing.assert_allclose(grad_c, grad_np, atol=1e-9)


@pytest.mark.parametrize("trend", ["add", "damped", None])
@pytest.mark.parametrize("seasonal_type", ["additive", "multiplicative"])
def test_fit_dispatches_to_compiled_backend_for_nan_containing_y(
    trend: str | None, seasonal_type: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """#36: a NaN-containing y must still use the compiled backend, not fall back to NumPy."""
    rng = np.random.default_rng(0)
    y = (
        rng.uniform(15, 25, 40)
        if seasonal_type == "multiplicative"
        else np.cumsum(rng.normal(0.5, 1.0, 40)) + 20
    )
    y[10] = np.nan
    calls = []
    real = _holt_hot.seasonal_objective_and_grad

    def spy(*args: object, **kwargs: object) -> tuple[float, np.ndarray]:
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(_holt_hot, "seasonal_objective_and_grad", spy)
    ExponentialSmoothing(
        h=3, trend=trend, seasonal_period=_M, seasonal_type=seasonal_type, n_starts=2, loss="mse"
    ).fit(y)
    assert calls, "fit() did not call the compiled backend for a NaN-containing y"


@pytest.mark.parametrize("kind", _KINDS)
def test_matches_numpy_reference_with_sample_weight(kind: str) -> None:
    """#39: the compiled seasonal core's sample_weight handling must match the NumPy path."""
    rng = np.random.default_rng(_SEED["mse", kind] + 300)
    multiplicative = kind in holt_module._MULTIPLICATIVE_KINDS
    y = rng.uniform(15, 25, 40) if multiplicative else np.cumsum(rng.normal(0.5, 1.0, 40)) + 20
    sample_weight = rng.uniform(0.1, 2.0, len(y))
    h = 5
    params = _random_params(kind, rng, multiplicative)
    loss_key = holt_module._loss_key("mse", _DELTA, _TAU, _NB_ALPHA, _TW_P)

    value_np, grad_np = holt_module._seasonal_objective_and_grad(
        params, y, h, kind, _M, loss_key, sample_weight=sample_weight
    )

    alpha, beta, phi, gamma, level0, trend0, season0 = holt_module._UNPACK_SEASONAL_BY_KIND[kind](
        params, _M
    )
    full = np.concatenate([[alpha, beta, phi, gamma, level0, trend0], season0])
    value_c, grad_c_full = _holt_hot.seasonal_objective_and_grad(
        full,
        y,
        h,
        _M,
        int(multiplicative),
        _LOSS_ID["mse"],
        _DELTA,
        _TAU,
        _NB_ALPHA,
        _TW_P,
        sample_weight,
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

    assert value_c == pytest.approx(value_np, abs=1e-9)
    np.testing.assert_allclose(grad_c, grad_np, atol=1e-9)


@pytest.mark.parametrize("trend", ["add", "damped", None])
@pytest.mark.parametrize("seasonal_type", ["additive", "multiplicative"])
def test_weighted_fit_dispatches_to_compiled_backend(
    trend: str | None, seasonal_type: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """#39: a weighted fit with a built-in loss must use the compiled backend, not fall back."""
    rng = np.random.default_rng(0)
    y = (
        rng.uniform(15, 25, 40)
        if seasonal_type == "multiplicative"
        else np.cumsum(rng.normal(0.5, 1.0, 40)) + 20
    )
    sample_weight = rng.uniform(0.1, 2.0, len(y))
    calls = []
    real = _holt_hot.seasonal_objective_and_grad

    def spy(*args: object, **kwargs: object) -> tuple[float, np.ndarray]:
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(_holt_hot, "seasonal_objective_and_grad", spy)
    ExponentialSmoothing(
        h=3, trend=trend, seasonal_period=_M, seasonal_type=seasonal_type, n_starts=2, loss="mse"
    ).fit(y, sample_weight=sample_weight)
    assert calls, "a weighted fit did not call the compiled backend"
