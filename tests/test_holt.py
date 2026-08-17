import numpy as np
import pytest
from statsmodels.tsa.holtwinters import Holt as SMHolt

from multi_ets import Holt


def test_predict_returns_h_forecasts():
    y = np.linspace(0, 10, 20)
    forecasts = Holt(h=4, loss="mse").fit(y).predict()
    assert np.shape(forecasts) == (4,)


def test_recovers_pure_linear_trend():
    y = 3.0 + 2.0 * np.arange(30)
    forecasts = Holt(h=5, loss="mse").fit(y).predict()
    expected = y[-1] + 2.0 * np.arange(1, 6)
    np.testing.assert_allclose(forecasts, expected, atol=1e-2)


def test_matches_statsmodels_classical_holt_at_h1():
    rng = np.random.default_rng(0)
    y = np.cumsum(rng.normal(0.5, 1.0, 50)) + 20

    sm_fit = SMHolt(y, initialization_method="estimated", damped_trend=False).fit(
        optimized=True
    )
    ours = Holt(h=1, loss="mse").fit(y)

    np.testing.assert_allclose(ours.predict(), sm_fit.forecast(1), atol=1e-3)


def test_too_short_series_raises():
    y = np.arange(3, dtype=float)
    with pytest.raises(ValueError, match="too short"):
        Holt(h=5, loss="mse").fit(y)


def test_multi_start_beats_single_start():
    # Flat then a sharp trend kink: a single fixed starting point lands in a
    # worse local optimum than the best of a 10-point restart grid, so its
    # held-out forecast is measurably worse.
    h = 3
    y_full = np.concatenate([np.full(15, 10.0), 10 + 5 * np.arange(1, 21)])
    y_fit, y_future = y_full[:-h], y_full[-h:]

    single = Holt(h=h, loss="mse", n_starts=1).fit(y_fit)
    multi = Holt(h=h, loss="mse", n_starts=10).fit(y_fit)

    err_single = np.abs(single.predict() - y_future).mean()
    err_multi = np.abs(multi.predict() - y_future).mean()
    assert err_multi < err_single
