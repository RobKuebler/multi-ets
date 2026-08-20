import jax.numpy as jnp
import numpy as np
import pytest
from statsmodels.tsa.holtwinters import Holt as SMHolt

from multi_ets import SES, DampedHolt, DampedHoltWinters, Holt, HoltWinters


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

    sm_fit = SMHolt(y, initialization_method="estimated", damped_trend=False).fit(optimized=True)
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


def test_mae_loss_gives_different_params_than_mse():
    rng = np.random.default_rng(1)
    y = 5.0 + 1.5 * np.arange(25) + rng.normal(0, 0.3, 25)
    mse_fit = Holt(h=3, loss="mse", n_starts=3).fit(y)
    mae_fit = Holt(h=3, loss="mae", n_starts=3).fit(y)
    assert not np.allclose(mse_fit.params_, mae_fit.params_)


def test_mae_is_less_pulled_by_outlier_than_mse():
    h = 3
    y = 5.0 + 1.5 * np.arange(30)
    y_outlier = y.copy()
    y_outlier[20] += 50  # spike well before the forecast origin

    clean_forecast = y[-1] + 1.5 * np.arange(1, h + 1)
    mse_fit = Holt(h=h, loss="mse", n_starts=5).fit(y_outlier)
    mae_fit = Holt(h=h, loss="mae", n_starts=5).fit(y_outlier)

    mse_err = np.abs(mse_fit.predict() - clean_forecast).mean()
    mae_err = np.abs(mae_fit.predict() - clean_forecast).mean()
    assert mae_err < mse_err


def test_huber_delta_measurably_affects_fit():
    h = 3
    y = 5.0 + 1.5 * np.arange(30)
    y_outlier = y.copy()
    y_outlier[20] += 50

    small_delta = Holt(h=h, loss="huber", delta=0.5, n_starts=5).fit(y_outlier)
    large_delta = Holt(h=h, loss="huber", delta=20.0, n_starts=5).fit(y_outlier)
    assert not np.allclose(small_delta.params_, large_delta.params_)


def test_pinball_tau_orders_forecasts():
    h = 3
    rng = np.random.default_rng(3)
    y = 5.0 + 1.5 * np.arange(40) + rng.normal(0, 1.5, 40)

    low_tau = Holt(h=h, loss="pinball", tau=0.1, n_starts=8).fit(y)
    high_tau = Holt(h=h, loss="pinball", tau=0.9, n_starts=8).fit(y)
    assert np.all(high_tau.predict() >= low_tau.predict())


def test_custom_callable_loss_is_used_to_fit():
    def weighted_mse(y_true, y_pred):
        return jnp.mean((y_pred - y_true) ** 2)

    y = 3.0 + 2.0 * np.arange(30)
    forecasts = Holt(h=5, loss=weighted_mse).fit(y).predict()
    expected = y[-1] + 2.0 * np.arange(1, 6)
    np.testing.assert_allclose(forecasts, expected, atol=1e-2)


def test_unknown_loss_raises():
    with pytest.raises(ValueError, match="Unknown loss"):
        Holt(h=3, loss="bogus")


def test_non_positive_huber_delta_raises():
    with pytest.raises(ValueError, match="delta must be positive"):
        Holt(h=3, loss="huber", delta=0.0)


def test_out_of_range_pinball_tau_raises():
    with pytest.raises(ValueError, match="tau must be in"):
        Holt(h=3, loss="pinball", tau=1.5)


def test_damped_predict_returns_h_forecasts():
    y = np.linspace(0, 10, 20)
    forecasts = DampedHolt(h=4, loss="mse").fit(y).predict()
    assert np.shape(forecasts) == (4,)


def test_damped_phi_stays_in_open_unit_interval():
    y = np.linspace(0, 10, 20)
    model = DampedHolt(h=4, loss="mse").fit(y)
    phi = model.params_[2]
    assert 0 < phi < 1


def test_damped_beats_undamped_on_trend_that_levels_off():
    h = 4
    t = np.arange(1, 31)
    true_phi = 0.6
    y_full = 10.0 + 8.0 * np.cumsum(true_phi**t)  # trend decaying to a flat asymptote
    y_fit, y_future = y_full[:-h], y_full[-h:]

    undamped = Holt(h=h, loss="mse", n_starts=10).fit(y_fit)
    damped = DampedHolt(h=h, loss="mse", n_starts=10).fit(y_fit)

    err_undamped = np.abs(undamped.predict() - y_future).mean()
    err_damped = np.abs(damped.predict() - y_future).mean()
    assert err_damped < err_undamped


def test_holt_winters_predict_returns_h_forecasts():
    y = np.linspace(0, 10, 20) + np.tile([1.0, -1.0, 2.0, -2.0], 5)
    forecasts = HoltWinters(h=4, seasonal_period=4, loss="mse").fit(y).predict()
    assert np.shape(forecasts) == (4,)


def test_damped_holt_winters_predict_returns_h_forecasts():
    y = np.linspace(0, 10, 20) + np.tile([1.0, -1.0, 2.0, -2.0], 5)
    forecasts = DampedHoltWinters(h=4, seasonal_period=4, loss="mse").fit(y).predict()
    assert np.shape(forecasts) == (4,)


def test_holt_winters_recovers_trend_and_seasonal_pattern():
    m, h = 4, 4
    t = np.arange(44)
    seasonal_true = np.array([5.0, -3.0, -4.0, 2.0])
    y_full = 20.0 + 1.5 * t + seasonal_true[t % m]
    y_fit, y_future = y_full[:-h], y_full[-h:]

    forecasts = HoltWinters(h=h, seasonal_period=m, loss="mse", n_starts=8).fit(y_fit).predict()
    np.testing.assert_allclose(forecasts, y_future, atol=0.5)


def test_holt_winters_too_short_series_raises():
    y = np.arange(5, dtype=float)
    with pytest.raises(ValueError, match="too short"):
        HoltWinters(h=2, seasonal_period=4).fit(y)


def test_holt_winters_seasonal_factors_mean_zero_and_rescale_invariant():
    m, h = 4, 4
    t = np.arange(44)
    seasonal_true = np.array([5.0, -3.0, -4.0, 2.0])
    y = 20.0 + 1.5 * t + seasonal_true[t % m]

    model = HoltWinters(h=h, seasonal_period=m, loss="mse", n_starts=8).fit(y)
    # params_ layout: alpha, beta, gamma, level0, trend0, s_0..s_{m-1}
    season = model.params_[5 : 5 + m]
    np.testing.assert_allclose(season.mean(), 0.0, atol=1e-6)

    forecasts = model.predict()
    c = 7.3  # arbitrary shift: -c into level0, +c into every seasonal factor
    model.params_[3] -= c
    model.params_[5 : 5 + m] += c
    np.testing.assert_allclose(model.predict(), forecasts, atol=1e-8)


def test_holt_winters_multiplicative_predict_returns_h_forecasts():
    m, h = 4, 4
    t = np.arange(20)
    seasonal_true = np.array([1.2, 0.8, 0.9, 1.1])
    y = (20.0 + 1.5 * t) * seasonal_true[t % m]
    forecasts = (
        HoltWinters(h=h, seasonal_period=m, seasonal_type="multiplicative", loss="mse")
        .fit(y)
        .predict()
    )
    assert np.shape(forecasts) == (h,)


def test_damped_holt_winters_multiplicative_predict_returns_h_forecasts():
    m, h = 4, 4
    t = np.arange(20)
    seasonal_true = np.array([1.2, 0.8, 0.9, 1.1])
    y = (20.0 + 1.5 * t) * seasonal_true[t % m]
    forecasts = (
        DampedHoltWinters(h=h, seasonal_period=m, seasonal_type="multiplicative", loss="mse")
        .fit(y)
        .predict()
    )
    assert np.shape(forecasts) == (h,)


def test_holt_winters_multiplicative_recovers_growing_seasonal_swing():
    # Seasonal swing is a fixed ratio of the level, so its absolute amplitude
    # grows as the level grows -- only a multiplicative model tracks this.
    m, h = 4, 4
    t = np.arange(44)
    seasonal_true = np.array([1.2, 0.8, 0.9, 1.1])
    level = 20.0 + 1.5 * t
    y_full = level * seasonal_true[t % m]
    y_fit, y_future = y_full[:-h], y_full[-h:]

    forecasts = (
        HoltWinters(h=h, seasonal_period=m, seasonal_type="multiplicative", loss="mse", n_starts=8)
        .fit(y_fit)
        .predict()
    )
    np.testing.assert_allclose(forecasts, y_future, atol=1.0)


def test_holt_winters_multiplicative_seasonal_factors_are_positive():
    m, h = 4, 4
    t = np.arange(44)
    seasonal_true = np.array([1.2, 0.8, 0.9, 1.1])
    y = (20.0 + 1.5 * t) * seasonal_true[t % m]

    model = HoltWinters(
        h=h, seasonal_period=m, seasonal_type="multiplicative", loss="mse", n_starts=8
    ).fit(y)
    season = model.params_[5 : 5 + m]
    assert np.all(season > 0)


def test_holt_winters_multiplicative_seasonal_factors_mean_one_and_rescale_invariant():
    m, h = 4, 4
    t = np.arange(44)
    seasonal_true = np.array([1.2, 0.8, 0.9, 1.1])
    y = (20.0 + 1.5 * t) * seasonal_true[t % m]

    model = HoltWinters(
        h=h, seasonal_period=m, seasonal_type="multiplicative", loss="mse", n_starts=8
    ).fit(y)
    # params_ layout: alpha, beta, gamma, level0, trend0, s_0..s_{m-1}
    season = model.params_[5 : 5 + m]
    np.testing.assert_allclose(season.mean(), 1.0, atol=1e-6)

    forecasts = model.predict()
    c = 1.7  # arbitrary scale: *c into level0/trend0, /c into every seasonal factor
    model.params_[3] *= c
    model.params_[4] *= c
    model.params_[5 : 5 + m] /= c
    np.testing.assert_allclose(model.predict(), forecasts, atol=1e-6)


def test_holt_winters_unknown_seasonal_type_raises():
    with pytest.raises(ValueError, match="Unknown seasonal_type"):
        HoltWinters(h=4, seasonal_period=4, seasonal_type="bogus")


def test_ses_predict_returns_h_forecasts():
    y = np.linspace(0, 10, 20)
    forecasts = SES(h=4, loss="mse").fit(y).predict()
    assert np.shape(forecasts) == (4,)


def test_ses_forecast_is_flat():
    y = np.linspace(0, 10, 20)
    forecasts = SES(h=4, loss="mse").fit(y).predict()
    np.testing.assert_allclose(forecasts, np.full(4, forecasts[0]))


def test_ses_recovers_constant_level():
    y = np.full(20, 7.0)
    forecasts = SES(h=3, loss="mse").fit(y).predict()
    np.testing.assert_allclose(forecasts, np.full(3, 7.0), atol=1e-2)


def test_ses_too_short_series_raises():
    y = np.arange(3, dtype=float)
    with pytest.raises(ValueError, match="too short"):
        SES(h=5, loss="mse").fit(y)


def test_predict_interval_contains_point_forecast():
    y = 3.0 + 2.0 * np.arange(30)
    model = Holt(h=5, loss="mse").fit(y)
    lower, upper = model.predict_interval(coverage=0.9)
    point = model.predict()
    assert np.all(lower <= point) and np.all(point <= upper)


def test_predict_interval_widens_with_coverage():
    rng = np.random.default_rng(2)
    y = 5.0 + 1.5 * np.arange(40) + rng.normal(0, 1.0, 40)
    model = Holt(h=4, loss="mse").fit(y)
    lower_narrow, upper_narrow = model.predict_interval(coverage=0.5)
    lower_wide, upper_wide = model.predict_interval(coverage=0.95)
    assert np.all((upper_wide - lower_wide) >= (upper_narrow - lower_narrow))


def test_predict_interval_list_of_coverages_returns_dict():
    y = 3.0 + 2.0 * np.arange(30)
    model = Holt(h=5, loss="mse").fit(y)
    result = model.predict_interval(coverage=[0.5, 0.9])
    assert set(result) == {0.5, 0.9}
    for lower, upper in result.values():
        assert np.shape(lower) == (5,)
        assert np.shape(upper) == (5,)


def test_predict_interval_works_for_seasonal_model():
    m, h = 4, 4
    t = np.arange(44)
    seasonal_true = np.array([5.0, -3.0, -4.0, 2.0])
    y = 20.0 + 1.5 * t + seasonal_true[t % m]
    model = HoltWinters(h=h, seasonal_period=m, loss="mse", n_starts=8).fit(y)
    lower, upper = model.predict_interval(coverage=0.9)
    point = model.predict()
    assert np.all(lower <= point) and np.all(point <= upper)


def test_predict_interval_works_for_ses():
    y = np.linspace(0, 10, 20)
    model = SES(h=4, loss="mse").fit(y)
    lower, upper = model.predict_interval(coverage=0.9)
    point = model.predict()
    assert np.all(lower <= point) and np.all(point <= upper)


def test_attribute_set_after_construction_is_honored_at_fit():
    y = 3.0 + 2.0 * np.arange(30)
    model = Holt(h=5, loss="huber", delta=1.0)
    model.delta = 20.0
    model.fit(y)  # must not raise, and must fit with delta=20.0, not the original 1.0

    direct = Holt(h=5, loss="huber", delta=20.0).fit(y)
    np.testing.assert_allclose(model.params_, direct.params_)
