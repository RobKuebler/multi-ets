import numpy as np
import pytest
from scipy.optimize import approx_fprime
from statsmodels.tsa.holtwinters import Holt as SMHolt

from multi_ets import ExponentialSmoothing
from multi_ets import holt as holt_module


def test_predict_returns_h_forecasts():
    y = np.linspace(0, 10, 20)
    forecasts = ExponentialSmoothing(trend="add", h=4, loss="mse").fit(y).predict()
    assert np.shape(forecasts) == (4,)


def test_recovers_pure_linear_trend():
    y = 3.0 + 2.0 * np.arange(30)
    forecasts = ExponentialSmoothing(trend="add", h=5, loss="mse").fit(y).predict()
    expected = y[-1] + 2.0 * np.arange(1, 6)
    np.testing.assert_allclose(forecasts, expected, atol=1e-2)


def test_matches_statsmodels_classical_holt_at_h1():
    rng = np.random.default_rng(0)
    y = np.cumsum(rng.normal(0.5, 1.0, 50)) + 20

    sm_fit = SMHolt(y, initialization_method="estimated", damped_trend=False).fit(optimized=True)
    ours = ExponentialSmoothing(trend="add", h=1, loss="mse").fit(y)

    np.testing.assert_allclose(ours.predict(), sm_fit.forecast(1), atol=1e-3)


def test_too_short_series_raises():
    y = np.arange(3, dtype=float)
    with pytest.raises(ValueError, match="too short"):
        ExponentialSmoothing(trend="add", h=5, loss="mse").fit(y)


def test_multi_start_beats_single_start():
    # Flat then a sharp trend kink: a single fixed starting point lands in a
    # worse local optimum than the best of a 10-point restart grid, so its
    # held-out forecast is measurably worse.
    h = 3
    y_full = np.concatenate([np.full(15, 10.0), 10 + 5 * np.arange(1, 21)])
    y_fit, y_future = y_full[:-h], y_full[-h:]

    single = ExponentialSmoothing(trend="add", h=h, loss="mse", n_starts=1).fit(y_fit)
    multi = ExponentialSmoothing(trend="add", h=h, loss="mse", n_starts=10).fit(y_fit)

    err_single = np.abs(single.predict() - y_future).mean()
    err_multi = np.abs(multi.predict() - y_future).mean()
    assert err_multi < err_single


def test_mae_loss_gives_different_params_than_mse():
    rng = np.random.default_rng(1)
    y = 5.0 + 1.5 * np.arange(25) + rng.normal(0, 0.3, 25)
    mse_fit = ExponentialSmoothing(trend="add", h=3, loss="mse", n_starts=3).fit(y)
    mae_fit = ExponentialSmoothing(trend="add", h=3, loss="mae", n_starts=3).fit(y)
    assert not np.allclose(mse_fit.params_, mae_fit.params_)


def test_mae_is_less_pulled_by_outlier_than_mse():
    h = 3
    y = 5.0 + 1.5 * np.arange(30)
    y_outlier = y.copy()
    y_outlier[20] += 50  # spike well before the forecast origin

    clean_forecast = y[-1] + 1.5 * np.arange(1, h + 1)
    mse_fit = ExponentialSmoothing(trend="add", h=h, loss="mse", n_starts=5).fit(y_outlier)
    mae_fit = ExponentialSmoothing(trend="add", h=h, loss="mae", n_starts=5).fit(y_outlier)

    mse_err = np.abs(mse_fit.predict() - clean_forecast).mean()
    mae_err = np.abs(mae_fit.predict() - clean_forecast).mean()
    assert mae_err < mse_err


def test_huber_delta_measurably_affects_fit():
    h = 3
    y = 5.0 + 1.5 * np.arange(30)
    y_outlier = y.copy()
    y_outlier[20] += 50

    small_delta = ExponentialSmoothing(trend="add", h=h, loss="huber", delta=0.5, n_starts=5).fit(
        y_outlier
    )
    large_delta = ExponentialSmoothing(trend="add", h=h, loss="huber", delta=20.0, n_starts=5).fit(
        y_outlier
    )
    assert not np.allclose(small_delta.params_, large_delta.params_)


def test_logcosh_is_less_pulled_by_outlier_than_mse():
    # log-cosh is a smooth Huber-like alternative (#30): it should be
    # measurably more robust to an outlier than mse, comparable to huber/mae
    # on the same contaminated series (mirrors test_mae_is_less_pulled_by_
    # outlier_than_mse and test_huber_delta_measurably_affects_fit above).
    h = 3
    y = 5.0 + 1.5 * np.arange(30)
    y_outlier = y.copy()
    y_outlier[20] += 50  # spike well before the forecast origin

    clean_forecast = y[-1] + 1.5 * np.arange(1, h + 1)
    mse_fit = ExponentialSmoothing(trend="add", h=h, loss="mse", n_starts=5).fit(y_outlier)
    huber_fit = ExponentialSmoothing(trend="add", h=h, loss="huber", n_starts=5).fit(y_outlier)
    logcosh_fit = ExponentialSmoothing(trend="add", h=h, loss="logcosh", n_starts=5).fit(y_outlier)

    mse_err = np.abs(mse_fit.predict() - clean_forecast).mean()
    huber_err = np.abs(huber_fit.predict() - clean_forecast).mean()
    logcosh_err = np.abs(logcosh_fit.predict() - clean_forecast).mean()

    assert logcosh_err < 0.5 * mse_err  # meaningfully more robust than mse
    assert logcosh_err < 2 * huber_err  # comparable in magnitude to huber's robustness


def test_logcosh_fits_and_predicts_with_no_extra_params():
    # Acceptance criterion (#30): loss="logcosh" alone is enough, no delta/tau/etc.
    y = 3.0 + 2.0 * np.arange(30)
    forecasts = ExponentialSmoothing(trend="add", h=5, loss="logcosh").fit(y).predict()
    assert np.shape(forecasts) == (5,)


def test_pinball_tau_orders_forecasts():
    h = 3
    rng = np.random.default_rng(3)
    y = 5.0 + 1.5 * np.arange(40) + rng.normal(0, 1.5, 40)

    low_tau = ExponentialSmoothing(trend="add", h=h, loss="pinball", tau=0.1, n_starts=8).fit(y)
    high_tau = ExponentialSmoothing(trend="add", h=h, loss="pinball", tau=0.9, n_starts=8).fit(y)
    assert np.all(high_tau.predict() >= low_tau.predict())


def test_expectile_tau_orders_forecasts():
    h = 3
    rng = np.random.default_rng(3)
    y = 5.0 + 1.5 * np.arange(40) + rng.normal(0, 1.5, 40)

    mid_tau = ExponentialSmoothing(trend="add", h=h, loss="expectile", tau=0.5, n_starts=8).fit(y)
    low_tau = ExponentialSmoothing(trend="add", h=h, loss="expectile", tau=0.1, n_starts=8).fit(y)
    high_tau = ExponentialSmoothing(trend="add", h=h, loss="expectile", tau=0.9, n_starts=8).fit(y)
    assert np.all(high_tau.predict() >= mid_tau.predict())
    assert np.all(mid_tau.predict() >= low_tau.predict())


def test_custom_callable_loss_is_used_to_fit():
    def weighted_mse(y_true, y_pred):
        return np.mean((y_pred - y_true) ** 2)

    y = 3.0 + 2.0 * np.arange(30)
    forecasts = ExponentialSmoothing(trend="add", h=5, loss=weighted_mse).fit(y).predict()
    expected = y[-1] + 2.0 * np.arange(1, 6)
    np.testing.assert_allclose(forecasts, expected, atol=1e-2)


@pytest.mark.parametrize(
    "loss_name,delta,tau",
    [
        ("mse", 1.0, 0.5),
        ("mae", 1.0, 0.5),
        ("huber", 0.7, 0.5),
        ("pinball", 1.0, 0.3),
        ("poisson", 1.0, 0.5),
        ("negbinom", 1.0, 0.5),
        ("tweedie", 1.0, 0.5),
        ("expectile", 1.0, 0.3),
        ("logcosh", 1.0, 0.5),
    ],
)
@pytest.mark.parametrize("kind,n_params", [("undamped", 4), ("damped", 5), ("ses", 2)])
@pytest.mark.parametrize("with_nan", [False, True])
def test_hand_derived_grad_matches_finite_difference(
    kind, n_params, loss_name, delta, tau, with_nan
):
    # ExponentialSmoothing.fit()'s hand-derived BPTT gradient (_holt_objective_and_grad) must
    # agree with a finite-difference approximation, for every built-in loss
    # and every non-seasonal kind (undamped/damped/ses). #18/#19 cross-checked
    # this against JAX autodiff while migrating off it; #20 drops jax as a
    # dependency entirely, so this is now the permanent regression check.
    rng = np.random.default_rng(0)
    h = 3
    y = np.cumsum(rng.normal(0, 1, 20)) + 10
    y = y - y.min() + 5.0  # poisson/negbinom/tweedie need non-negative y; don't rely on luck
    if with_nan:
        y[7:9] = np.nan  # gradient through skipped (missing) steps
    loss_key = (loss_name, delta, tau, 1.0, 1.5)

    p = rng.uniform(0.1, 0.9, n_params)
    if kind == "damped":
        p[2] = min(p[2], 0.9)  # phi < 1, matching the classical damping range

    def value(params):
        return holt_module._holt_objective_and_grad(params, y, h, kind, loss_key)[0]

    _, grad = holt_module._holt_objective_and_grad(p, y, h, kind, loss_key)
    numeric_grad = approx_fprime(p, value, 1e-6)

    np.testing.assert_allclose(grad, numeric_grad, atol=1e-4, rtol=1e-4)


def test_hand_derived_grad_matches_finite_difference_with_sample_weight():
    # #25: sample_weight feeds into the same hand-derived BPTT gradient, so it
    # needs its own finite-difference cross-check, with non-uniform weights
    # (all-ones would not exercise the weighting math at all).
    rng = np.random.default_rng(4)
    h, kind, n_params = 3, "undamped", 4
    y = np.cumsum(rng.normal(0, 1, 20)) + 10
    loss_key = ("mse", 1.0, 0.5, 1.0, 1.5)
    sample_weight = rng.uniform(0.1, 2.0, len(y))
    p = rng.uniform(0.1, 0.9, n_params)

    def value(params):
        return holt_module._holt_objective_and_grad(
            params, y, h, kind, loss_key, sample_weight=sample_weight
        )[0]

    _, grad = holt_module._holt_objective_and_grad(
        p, y, h, kind, loss_key, sample_weight=sample_weight
    )
    numeric_grad = approx_fprime(p, value, 1e-6)

    np.testing.assert_allclose(grad, numeric_grad, atol=1e-4, rtol=1e-4)


def _rand_seasonal_params(rng, kind, m, multiplicative):
    season0 = rng.uniform(0.5, 1.5, m) if multiplicative else rng.uniform(-1.0, 1.0, m)
    level0, trend0 = rng.uniform(-1.0, 1.0, 2)
    if kind.startswith("seasonal_no_trend"):
        alpha, gamma = rng.uniform(0.1, 0.9, 2)
        return np.concatenate([[alpha, gamma, level0], season0])
    if kind.startswith("damped"):
        alpha, beta, gamma = rng.uniform(0.1, 0.9, 3)
        phi = rng.uniform(0.1, holt_module._PHI_UPPER)  # exercise up to the classical damping cap
        return np.concatenate([[alpha, beta, phi, gamma, level0, trend0], season0])
    alpha, beta, gamma = rng.uniform(0.1, 0.9, 3)
    return np.concatenate([[alpha, beta, gamma, level0, trend0], season0])


@pytest.mark.parametrize(
    "loss_name,delta,tau",
    [
        ("mse", 1.0, 0.5),
        ("mae", 1.0, 0.5),
        ("huber", 0.7, 0.5),
        ("pinball", 1.0, 0.3),
        ("poisson", 1.0, 0.5),
        ("negbinom", 1.0, 0.5),
        ("tweedie", 1.0, 0.5),
        ("expectile", 1.0, 0.3),
        ("logcosh", 1.0, 0.5),
    ],
)
@pytest.mark.parametrize(
    "kind",
    [
        "seasonal",
        "damped_seasonal",
        "seasonal_no_trend",
        "seasonal_mult",
        "damped_seasonal_mult",
        "seasonal_no_trend_mult",
    ],
)
@pytest.mark.parametrize("with_nan", [False, True])
def test_seasonal_hand_derived_grad_matches_finite_difference(
    kind, loss_name, delta, tau, with_nan
):
    # ExponentialSmoothing.fit()'s hand-derived seasonal BPTT
    # gradient (_seasonal_objective_and_grad) must agree with a
    # finite-difference approximation, for every built-in loss,
    # additive+multiplicative, undamped+damped (see the non-seasonal
    # counterpart above for why this replaced a JAX-autodiff cross-check).
    rng = np.random.default_rng(2)
    h = 3
    m = 4
    multiplicative = kind in holt_module._MULTIPLICATIVE_KINDS
    y = np.cumsum(rng.normal(0, 1, 20)) + 10
    # Always keep y positive: multiplicative kinds need it for y_t/level, y_t/s_old, and
    # poisson/negbinom/tweedie need a non-negative y regardless of kind -- don't rely on
    # the unshifted random walk happening to stay positive for those rows.
    y = y - y.min() + 5.0
    if with_nan:
        y[7:9] = np.nan  # gradient through skipped (missing) steps
    loss_key = (loss_name, delta, tau, 1.0, 1.5)
    p = _rand_seasonal_params(rng, kind, m, multiplicative)

    def value(params):
        return holt_module._seasonal_objective_and_grad(params, y, h, kind, m, loss_key)[0]

    _, grad = holt_module._seasonal_objective_and_grad(p, y, h, kind, m, loss_key)
    numeric_grad = approx_fprime(p, value, 1e-6)

    np.testing.assert_allclose(grad, numeric_grad, atol=1e-4, rtol=1e-4)


def test_seasonal_hand_derived_grad_matches_finite_difference_with_sample_weight():
    # #25, seasonal counterpart of the non-seasonal weighted-gradient check above.
    rng = np.random.default_rng(5)
    h, m, kind = 3, 4, "seasonal"
    y = np.cumsum(rng.normal(0, 1, 20)) + 10
    loss_key = ("mse", 1.0, 0.5, 1.0, 1.5)
    sample_weight = rng.uniform(0.1, 2.0, len(y))
    p = _rand_seasonal_params(rng, kind, m, multiplicative=False)

    def value(params):
        return holt_module._seasonal_objective_and_grad(
            params, y, h, kind, m, loss_key, sample_weight=sample_weight
        )[0]

    _, grad = holt_module._seasonal_objective_and_grad(
        p, y, h, kind, m, loss_key, sample_weight=sample_weight
    )
    numeric_grad = approx_fprime(p, value, 1e-6)

    np.testing.assert_allclose(grad, numeric_grad, atol=1e-4, rtol=1e-4)


def test_unknown_loss_raises():
    with pytest.raises(ValueError, match="Unknown loss"):
        ExponentialSmoothing(trend="add", h=3, loss="bogus")


def test_non_positive_huber_delta_raises():
    with pytest.raises(ValueError, match="delta must be positive"):
        ExponentialSmoothing(trend="add", h=3, loss="huber", delta=0.0)


def test_out_of_range_pinball_tau_raises():
    with pytest.raises(ValueError, match="tau must be in"):
        ExponentialSmoothing(trend="add", h=3, loss="pinball", tau=1.5)


def test_poisson_loss_fits_on_count_like_series():
    # Smoke test: every non-seasonal/seasonal kind fits successfully with
    # loss="poisson" on a non-negative, count-like series (#26).
    rng = np.random.default_rng(8)
    y = rng.poisson(lam=20.0, size=30).astype(float)
    m, h = 4, 3

    ExponentialSmoothing(trend="add", h=h, loss="poisson", n_starts=3).fit(y)
    ExponentialSmoothing(h=h, loss="poisson", n_starts=3).fit(y)
    ExponentialSmoothing(trend="damped", h=h, loss="poisson", n_starts=3).fit(y)
    ExponentialSmoothing(trend="add", h=h, seasonal_period=m, loss="poisson", n_starts=3).fit(y)


def test_poisson_loss_rejects_negative_y():
    y = np.array([1.0, 2.0, -1.0, 3.0, 4.0, 5.0, 6.0])
    with pytest.raises(ValueError, match="non-negative"):
        ExponentialSmoothing(trend="add", h=3, loss="poisson").fit(y)


def test_poisson_loss_gives_lower_poisson_deviance_than_mse():
    # Core acceptance check (#26): fitting with the matching loss should beat
    # fitting with a mismatched one, evaluated on that loss's own scale.
    rng = np.random.default_rng(9)
    h = 3
    t = np.arange(40)
    lam = 5.0 + 0.5 * t
    y = rng.poisson(lam=lam).astype(float)

    poisson_fit = ExponentialSmoothing(trend="add", h=h, loss="poisson", n_starts=5).fit(y)
    mse_fit = ExponentialSmoothing(trend="add", h=h, loss="mse", n_starts=5).fit(y)

    poisson_loss_key = ("poisson", 1.0, 0.5, 1.0, 1.5)

    def poisson_deviance(model):
        *_, forecasts, actuals = holt_module._holt_forward_np(model.params_, y, h, model._kind)
        return holt_module._loss_value_np(poisson_loss_key, actuals, forecasts)

    assert poisson_deviance(poisson_fit) < poisson_deviance(mse_fit)


def test_out_of_range_expectile_tau_raises():
    with pytest.raises(ValueError, match="tau must be in"):
        ExponentialSmoothing(trend="add", h=3, loss="expectile", tau=1.5)


def test_damped_predict_returns_h_forecasts():
    y = np.linspace(0, 10, 20)
    forecasts = ExponentialSmoothing(trend="damped", h=4, loss="mse").fit(y).predict()
    assert np.shape(forecasts) == (4,)


def test_damped_phi_stays_in_open_unit_interval():
    y = np.linspace(0, 10, 20)
    model = ExponentialSmoothing(trend="damped", h=4, loss="mse").fit(y)
    phi = model.params_[2]
    assert 0 < phi < 1


def test_damped_phi_capped_on_strongly_trending_series():
    # Superlinear growth pins phi against the upper bound; it must stop at
    # the standard classical damping cap (0.98), not collapse to undamped (~1).
    y = 100.0 + np.arange(40, dtype=float) ** 2
    model = ExponentialSmoothing(trend="damped", h=4, loss="mse", n_starts=10).fit(y)
    phi = model.params_[2]
    assert phi <= 0.98


def test_damped_holt_winters_phi_capped_on_strongly_trending_series():
    m = 4
    t = np.arange(48)
    y = 100.0 + t.astype(float) ** 2 + np.tile([5.0, -3.0, -4.0, 2.0], 12)
    model = ExponentialSmoothing(
        trend="damped", h=4, seasonal_period=m, loss="mse", n_starts=10
    ).fit(y)
    phi = model.params_[2]
    assert phi <= 0.98


def test_damped_beats_undamped_on_trend_that_levels_off():
    h = 4
    t = np.arange(1, 31)
    true_phi = 0.6
    y_full = 10.0 + 8.0 * np.cumsum(true_phi**t)  # trend decaying to a flat asymptote
    y_fit, y_future = y_full[:-h], y_full[-h:]

    undamped = ExponentialSmoothing(trend="add", h=h, loss="mse", n_starts=10).fit(y_fit)
    damped = ExponentialSmoothing(trend="damped", h=h, loss="mse", n_starts=10).fit(y_fit)

    err_undamped = np.abs(undamped.predict() - y_future).mean()
    err_damped = np.abs(damped.predict() - y_future).mean()
    assert err_damped < err_undamped


def test_holt_winters_predict_returns_h_forecasts():
    y = np.linspace(0, 10, 20) + np.tile([1.0, -1.0, 2.0, -2.0], 5)
    forecasts = (
        ExponentialSmoothing(trend="add", h=4, seasonal_period=4, loss="mse").fit(y).predict()
    )
    assert np.shape(forecasts) == (4,)


def test_damped_holt_winters_predict_returns_h_forecasts():
    y = np.linspace(0, 10, 20) + np.tile([1.0, -1.0, 2.0, -2.0], 5)
    forecasts = (
        ExponentialSmoothing(trend="damped", h=4, seasonal_period=4, loss="mse").fit(y).predict()
    )
    assert np.shape(forecasts) == (4,)


def test_holt_winters_recovers_trend_and_seasonal_pattern():
    m, h = 4, 4
    t = np.arange(44)
    seasonal_true = np.array([5.0, -3.0, -4.0, 2.0])
    y_full = 20.0 + 1.5 * t + seasonal_true[t % m]
    y_fit, y_future = y_full[:-h], y_full[-h:]

    forecasts = (
        ExponentialSmoothing(trend="add", h=h, seasonal_period=m, loss="mse", n_starts=8)
        .fit(y_fit)
        .predict()
    )
    np.testing.assert_allclose(forecasts, y_future, atol=0.5)


def test_holt_winters_too_short_series_raises():
    y = np.arange(5, dtype=float)
    with pytest.raises(ValueError, match="too short"):
        ExponentialSmoothing(trend="add", h=2, seasonal_period=4).fit(y)


def test_holt_winters_seasonal_factors_mean_zero_and_rescale_invariant():
    m, h = 4, 4
    t = np.arange(44)
    seasonal_true = np.array([5.0, -3.0, -4.0, 2.0])
    y = 20.0 + 1.5 * t + seasonal_true[t % m]

    model = ExponentialSmoothing(trend="add", h=h, seasonal_period=m, loss="mse", n_starts=8).fit(y)
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
        ExponentialSmoothing(
            trend="add", h=h, seasonal_period=m, seasonal_type="multiplicative", loss="mse"
        )
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
        ExponentialSmoothing(
            trend="damped", h=h, seasonal_period=m, seasonal_type="multiplicative", loss="mse"
        )
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
        ExponentialSmoothing(
            trend="add",
            h=h,
            seasonal_period=m,
            seasonal_type="multiplicative",
            loss="mse",
            n_starts=8,
        )
        .fit(y_fit)
        .predict()
    )
    np.testing.assert_allclose(forecasts, y_future, atol=1.0)


def test_holt_winters_multiplicative_seasonal_factors_are_positive():
    m, h = 4, 4
    t = np.arange(44)
    seasonal_true = np.array([1.2, 0.8, 0.9, 1.1])
    y = (20.0 + 1.5 * t) * seasonal_true[t % m]

    model = ExponentialSmoothing(
        trend="add", h=h, seasonal_period=m, seasonal_type="multiplicative", loss="mse", n_starts=8
    ).fit(y)
    season = model.params_[5 : 5 + m]
    assert np.all(season > 0)


def test_holt_winters_multiplicative_seasonal_factors_mean_one_and_rescale_invariant():
    m, h = 4, 4
    t = np.arange(44)
    seasonal_true = np.array([1.2, 0.8, 0.9, 1.1])
    y = (20.0 + 1.5 * t) * seasonal_true[t % m]

    model = ExponentialSmoothing(
        trend="add", h=h, seasonal_period=m, seasonal_type="multiplicative", loss="mse", n_starts=8
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
        ExponentialSmoothing(trend="add", h=4, seasonal_period=4, seasonal_type="bogus")


# --- Seasonal without a trend term (#38) ---


def test_seasonal_no_trend_predict_returns_h_forecasts():
    m, h = 4, 4
    t = np.arange(20)
    y = 20.0 + np.tile([1.0, -1.0, 2.0, -2.0], 5)[t]
    forecasts = ExponentialSmoothing(h=h, seasonal_period=m, loss="mse").fit(y).predict()
    assert np.shape(forecasts) == (h,)


def test_seasonal_no_trend_kind_defaults_to_no_trend():
    model = ExponentialSmoothing(h=4, seasonal_period=4)
    assert model.trend is None
    assert model._kind == "seasonal_no_trend"


def test_seasonal_no_trend_recovers_flat_level_and_seasonal_pattern():
    m, h = 4, 4
    t = np.arange(44)
    seasonal_true = np.array([5.0, -3.0, -4.0, 2.0])
    y_full = 20.0 + seasonal_true[t % m]
    y_fit, y_future = y_full[:-h], y_full[-h:]

    forecasts = (
        ExponentialSmoothing(h=h, seasonal_period=m, loss="mse", n_starts=8).fit(y_fit).predict()
    )
    np.testing.assert_allclose(forecasts, y_future, atol=0.5)


def test_seasonal_no_trend_multiplicative_recovers_seasonal_pattern():
    m, h = 4, 4
    t = np.arange(44)
    seasonal_true = np.array([1.2, 0.8, 0.9, 1.1])
    y_full = 20.0 * seasonal_true[t % m]
    y_fit, y_future = y_full[:-h], y_full[-h:]

    forecasts = (
        ExponentialSmoothing(
            h=h, seasonal_period=m, seasonal_type="multiplicative", loss="mse", n_starts=8
        )
        .fit(y_fit)
        .predict()
    )
    np.testing.assert_allclose(forecasts, y_future, atol=0.5)


def test_seasonal_no_trend_seasonal_factors_mean_zero_and_rescale_invariant():
    m, h = 4, 4
    t = np.arange(20)
    seasonal_true = np.array([5.0, -3.0, -4.0, 2.0])
    y = 20.0 + seasonal_true[t % m]

    model = ExponentialSmoothing(h=h, seasonal_period=m, loss="mse", n_starts=8).fit(y)
    # params_ layout: alpha, gamma, level0, s_0..s_{m-1}
    season = model.params_[3 : 3 + m]
    np.testing.assert_allclose(season.mean(), 0.0, atol=1e-6)

    forecasts = model.predict()
    c = 7.3  # arbitrary shift: -c into level0, +c into every seasonal factor
    model.params_[2] -= c
    model.params_[3 : 3 + m] += c
    np.testing.assert_allclose(model.predict(), forecasts, atol=1e-8)


def test_seasonal_no_trend_multiplicative_seasonal_factors_mean_one():
    m, h = 4, 4
    t = np.arange(20)
    seasonal_true = np.array([1.2, 0.8, 0.9, 1.1])
    y = 20.0 * seasonal_true[t % m]

    model = ExponentialSmoothing(
        h=h, seasonal_period=m, seasonal_type="multiplicative", loss="mse", n_starts=8
    ).fit(y)
    season = model.params_[3 : 3 + m]
    np.testing.assert_allclose(season.mean(), 1.0, atol=1e-6)
    assert np.all(season > 0)


def test_seasonal_no_trend_predict_interval_works():
    m, h = 4, 4
    t = np.arange(20)
    y = 20.0 + np.tile([5.0, -3.0, -4.0, 2.0], 5)[t]
    model = ExponentialSmoothing(h=h, seasonal_period=m, loss="mse", n_starts=8).fit(y)
    lower, upper = model.predict_interval(coverage=0.9)
    point = model.predict()
    assert np.all(lower <= point) and np.all(point <= upper)


def test_seasonal_no_trend_fits_with_interior_nan_gap():
    m, h = 4, 4
    t = np.arange(44)
    seasonal_true = np.array([5.0, -3.0, -4.0, 2.0])
    y_full = 20.0 + seasonal_true[t % m]
    y_fit, y_future = y_full[:-h], y_full[-h:]
    y_gap = y_fit.copy()
    y_gap[10] = np.nan

    forecasts = (
        ExponentialSmoothing(h=h, seasonal_period=m, loss="mse", n_starts=8).fit(y_gap).predict()
    )
    np.testing.assert_allclose(forecasts, y_future, atol=1.0)


def test_trend_none_with_seasonal_period_no_longer_raises():
    # Regression guard for #38: this combination used to be rejected outright.
    model = ExponentialSmoothing(h=3, trend=None, seasonal_period=4)
    assert model._kind in ("seasonal_no_trend", "seasonal_no_trend_mult")


def test_ses_predict_returns_h_forecasts():
    y = np.linspace(0, 10, 20)
    forecasts = ExponentialSmoothing(h=4, loss="mse").fit(y).predict()
    assert np.shape(forecasts) == (4,)


def test_ses_forecast_is_flat():
    y = np.linspace(0, 10, 20)
    forecasts = ExponentialSmoothing(h=4, loss="mse").fit(y).predict()
    np.testing.assert_allclose(forecasts, np.full(4, forecasts[0]))


def test_ses_recovers_constant_level():
    y = np.full(20, 7.0)
    forecasts = ExponentialSmoothing(h=3, loss="mse").fit(y).predict()
    np.testing.assert_allclose(forecasts, np.full(3, 7.0), atol=1e-2)


def test_ses_too_short_series_raises():
    y = np.arange(3, dtype=float)
    with pytest.raises(ValueError, match="too short"):
        ExponentialSmoothing(h=5, loss="mse").fit(y)


def test_predict_interval_contains_point_forecast():
    y = 3.0 + 2.0 * np.arange(30)
    model = ExponentialSmoothing(trend="add", h=5, loss="mse").fit(y)
    lower, upper = model.predict_interval(coverage=0.9)
    point = model.predict()
    assert np.all(lower <= point) and np.all(point <= upper)


def test_predict_interval_widens_with_coverage():
    rng = np.random.default_rng(2)
    y = 5.0 + 1.5 * np.arange(40) + rng.normal(0, 1.0, 40)
    model = ExponentialSmoothing(trend="add", h=4, loss="mse").fit(y)
    lower_narrow, upper_narrow = model.predict_interval(coverage=0.5)
    lower_wide, upper_wide = model.predict_interval(coverage=0.95)
    assert np.all((upper_wide - lower_wide) >= (upper_narrow - lower_narrow))


def test_predict_interval_list_of_coverages_returns_dict():
    y = 3.0 + 2.0 * np.arange(30)
    model = ExponentialSmoothing(trend="add", h=5, loss="mse").fit(y)
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
    model = ExponentialSmoothing(trend="add", h=h, seasonal_period=m, loss="mse", n_starts=8).fit(y)
    lower, upper = model.predict_interval(coverage=0.9)
    point = model.predict()
    assert np.all(lower <= point) and np.all(point <= upper)


def test_predict_interval_works_for_ses():
    y = np.linspace(0, 10, 20)
    model = ExponentialSmoothing(h=4, loss="mse").fit(y)
    lower, upper = model.predict_interval(coverage=0.9)
    point = model.predict()
    assert np.all(lower <= point) and np.all(point <= upper)


def test_attribute_set_after_construction_is_honored_at_fit():
    y = 3.0 + 2.0 * np.arange(30)
    model = ExponentialSmoothing(trend="add", h=5, loss="huber", delta=1.0)
    model.delta = 20.0
    model.fit(y)  # must not raise, and must fit with delta=20.0, not the original 1.0

    direct = ExponentialSmoothing(trend="add", h=5, loss="huber", delta=20.0).fit(y)
    np.testing.assert_allclose(model.params_, direct.params_)


@pytest.mark.parametrize(
    "loss,kwargs",
    [
        ("mse", {}),
        ("mae", {}),
        ("huber", {"delta": 0.8}),
        ("pinball", {"tau": 0.4}),
        ("poisson", {}),
        ("negbinom", {"alpha": 2.0}),
        ("tweedie", {"p": 1.3}),
        ("expectile", {"tau": 0.3}),
        ("logcosh", {}),
    ],
)
def test_sample_weight_all_ones_matches_unweighted_fit(loss, kwargs):
    rng = np.random.default_rng(6)
    y = 5.0 + 1.2 * np.arange(25) + rng.normal(0, 0.5, 25)
    unweighted = ExponentialSmoothing(trend="add", h=3, loss=loss, n_starts=3, **kwargs).fit(y)
    weighted = ExponentialSmoothing(trend="add", h=3, loss=loss, n_starts=3, **kwargs).fit(
        y, sample_weight=np.ones(len(y))
    )
    np.testing.assert_allclose(weighted.params_, unweighted.params_, atol=1e-4)


def test_sample_weight_omitted_matches_unweighted_fit():
    y = 5.0 + 1.2 * np.arange(25)
    weighted = ExponentialSmoothing(trend="add", h=3, loss="mse", n_starts=3).fit(
        y, sample_weight=None
    )
    unweighted = ExponentialSmoothing(trend="add", h=3, loss="mse", n_starts=3).fit(y)
    np.testing.assert_allclose(weighted.params_, unweighted.params_)


def test_holt_winters_sample_weight_all_ones_matches_unweighted_fit():
    rng = np.random.default_rng(7)
    m = 4
    y = 20 + 0.5 * np.arange(40) + 5 * np.sin(2 * np.pi * np.arange(40) / m)
    y += rng.normal(0, 0.2, 40)
    unweighted = ExponentialSmoothing(trend="add", h=3, seasonal_period=m, n_starts=3).fit(y)
    weighted = ExponentialSmoothing(trend="add", h=3, seasonal_period=m, n_starts=3).fit(
        y, sample_weight=np.ones(40)
    )
    np.testing.assert_allclose(weighted.params_, unweighted.params_, atol=1e-4)


def test_sample_weight_wrong_length_raises():
    y = np.arange(20, dtype=float)
    with pytest.raises(ValueError, match="sample_weight"):
        ExponentialSmoothing(trend="add", h=3, loss="mse").fit(y, sample_weight=np.ones(10))


def test_sample_weight_with_custom_callable_loss_raises():
    def mse(y_true, y_pred):
        return np.mean((y_pred - y_true) ** 2)

    y = np.arange(20, dtype=float)
    with pytest.raises(ValueError, match="custom callable"):
        ExponentialSmoothing(trend="add", h=3, loss=mse).fit(y, sample_weight=np.ones(20))


def test_sample_weight_negative_raises():
    y = np.arange(20, dtype=float)
    weight = np.ones(20)
    weight[5] = -1.0
    with pytest.raises(ValueError, match="non-negative"):
        ExponentialSmoothing(trend="add", h=3, loss="mse").fit(y, sample_weight=weight)


def test_sample_weight_all_zero_raises():
    y = np.arange(20, dtype=float)
    with pytest.raises(ValueError, match="positive weight"):
        ExponentialSmoothing(trend="add", h=3, loss="mse").fit(y, sample_weight=np.zeros(20))


def test_sample_weight_that_only_covers_a_nan_gap_raises():
    # Passes _resolve_sample_weight's upfront check (weight.sum() > 0), but
    # every bit of positive weight lands on the one NaN actual -- so after
    # NaN-masking, the effective total weight is 0. Must raise a clear error
    # instead of silently returning a NaN loss/gradient (code review finding
    # on #39: this is reachable through both the NumPy and compiled paths).
    y = np.array([5.0, 3.0, np.nan])
    sample_weight = np.array([0.0, 0.0, 1.0])
    with pytest.raises(ValueError, match="no positive-weight"):
        ExponentialSmoothing(trend="add", h=1, n_starts=2, loss="mse").fit(
            y, sample_weight=sample_weight
        )


def test_weight_matrix_aligns_with_actuals_construction():
    # weight[t, k] must line up with actuals's np.stack([y[t:t+h] ...]) layout,
    # so an observation is weighted the same in every origin window it falls
    # into, not just the one where it's the origin.
    sample_weight = np.arange(10, dtype=float)
    n_origins, h = 4, 3
    w = holt_module._weight_matrix_np(sample_weight, n_origins, h)
    for t in range(n_origins):
        np.testing.assert_array_equal(w[t], sample_weight[t : t + h])


def test_sample_weight_zero_excludes_observation_from_every_origin_window():
    # Same observation appears at (t, k) whenever t + k == target_index;
    # zeroing its weight must drop all of those cells, not just the one
    # where it's the origin (t == target_index, k == 0).
    n_origins, h, target_index = 4, 3, 2
    actuals = np.arange(n_origins * h, dtype=float).reshape(n_origins, h)
    forecasts = np.zeros_like(actuals)
    sample_weight = np.ones(n_origins + h - 1)
    sample_weight[target_index] = 0.0
    weight = holt_module._weight_matrix_np(sample_weight, n_origins, h)

    value = holt_module._loss_value_np(
        ("mse", 1.0, 0.5, 1.0, 1.5), actuals, forecasts, weight=weight
    )

    kept = actuals[weight > 0]
    assert value == pytest.approx(float((kept**2).mean()))


def test_masked_mean_zero_weight_ignores_inf_term():
    # A zero-weighted term must not poison the mean even if its value is inf
    # (0 * inf = nan under plain multiplication).
    x = np.array([[1.0, 2.0], [np.inf, np.inf], [3.0, 4.0]])
    weight = np.array([[1.0, 1.0], [0.0, 0.0], [1.0, 1.0]])
    result = holt_module._masked_mean_np(x, mask=None, weight=weight)
    assert result == pytest.approx((1 + 2 + 3 + 4) / 4)


def test_sample_weight_downweighting_outlier_improves_fit():
    # Same spike as test_mae_is_less_pulled_by_outlier_than_mse, but instead of
    # switching loss, downweight the spiked observation directly: the weighted
    # fit should recover the clean trend better than the unweighted fit that
    # has to explain the spike too.
    h = 3
    y = 5.0 + 1.5 * np.arange(30)
    y_outlier = y.copy()
    y_outlier[20] += 50
    clean_forecast = y[-1] + 1.5 * np.arange(1, h + 1)

    weight = np.ones(30)
    weight[20] = 0.01

    unweighted = ExponentialSmoothing(trend="add", h=h, loss="mse", n_starts=5).fit(y_outlier)
    weighted = ExponentialSmoothing(trend="add", h=h, loss="mse", n_starts=5).fit(
        y_outlier, sample_weight=weight
    )

    assert not np.allclose(weighted.params_, unweighted.params_)
    err_unweighted = np.abs(unweighted.predict() - clean_forecast).mean()
    err_weighted = np.abs(weighted.predict() - clean_forecast).mean()
    assert err_weighted < err_unweighted


def test_negbinom_smoke_fits_for_all_model_kinds():
    # #27: negbinom must be usable end to end (not just differentiable) by
    # every non-seasonal and seasonal model kind.
    rng = np.random.default_rng(9)
    y = rng.poisson(10, size=30).astype(float)

    assert ExponentialSmoothing(trend="add", h=3, loss="negbinom", alpha=1.0).fit(
        y
    ).predict().shape == (3,)
    assert ExponentialSmoothing(h=3, loss="negbinom", alpha=1.0).fit(y).predict().shape == (3,)
    assert ExponentialSmoothing(trend="damped", h=3, loss="negbinom", alpha=1.0).fit(
        y
    ).predict().shape == (3,)

    y_seasonal = rng.poisson(10, size=40).astype(float)
    forecasts = (
        ExponentialSmoothing(trend="add", h=3, seasonal_period=4, loss="negbinom", alpha=1.0)
        .fit(y_seasonal)
        .predict()
    )
    assert forecasts.shape == (3,)


def test_non_positive_negbinom_alpha_raises():
    with pytest.raises(ValueError, match="alpha"):
        ExponentialSmoothing(trend="add", h=3, loss="negbinom", alpha=0.0)
    with pytest.raises(ValueError, match="alpha"):
        ExponentialSmoothing(trend="add", h=3, loss="negbinom", alpha=-1.0)


def test_negbinom_negative_y_raises():
    y = np.array([1.0, 2.0, -1.0, 3.0, 4.0, 5.0])
    with pytest.raises(ValueError, match="non-negative"):
        ExponentialSmoothing(trend="add", h=2, loss="negbinom", alpha=1.0).fit(y)


def test_negbinom_achieves_lower_negbinom_deviance_than_poisson_on_overdispersed_counts():
    # Overdispersed count series (variance well above the mean): a model
    # fit by minimizing Poisson deviance (which assumes variance == mean)
    # gets pulled differently by the extra variance than one that accounts
    # for it via negbinom's alpha, so under negbinom's own deviance the
    # negbinom-fit model should score better. loss="poisson" isn't
    # implemented in this worktree yet (#26, sibling issue), so the
    # comparison model is fit against the documented Poisson deviance
    # formula directly, via a custom callable loss.
    rng = np.random.default_rng(8)
    h = 3
    mu, alpha_true = 30.0, 0.3
    p = 1.0 / (1.0 + alpha_true * mu)
    n = mu * p / (1.0 - p)
    y = rng.negative_binomial(n, p, size=40).astype(float)

    def poisson_deviance(y_true, y_pred):
        mu_pred = np.maximum(y_pred, holt_module._BOUND_EPS)
        dev = 2.0 * (
            np.where(y_true > 0, y_true * np.log(y_true / mu_pred), 0.0) - (y_true - mu_pred)
        )
        return dev.mean()

    negbinom_fit = ExponentialSmoothing(
        trend="add", h=h, loss="negbinom", alpha=alpha_true, n_starts=5
    ).fit(y)
    poisson_fit = ExponentialSmoothing(trend="add", h=h, loss=poisson_deviance, n_starts=5).fit(y)

    *_, forecasts_nb, actuals_nb = holt_module._holt_forward_np(
        negbinom_fit.params_, y, h, negbinom_fit._kind
    )
    *_, forecasts_pois, actuals_pois = holt_module._holt_forward_np(
        poisson_fit.params_, y, h, poisson_fit._kind
    )

    loss_key = ("negbinom", 1.0, 0.5, alpha_true, 1.5)
    dev_nb = holt_module._loss_value_np(loss_key, actuals_nb, forecasts_nb)
    dev_pois = holt_module._loss_value_np(loss_key, actuals_pois, forecasts_pois)

    assert dev_nb < dev_pois


def test_tweedie_smoke_fits_all_kinds():
    # #28: loss="tweedie" must fit successfully (finite params/predictions) on
    # a non-negative series, for every model class.
    y = 5.0 + 1.2 * np.arange(25)
    assert np.all(
        np.isfinite(ExponentialSmoothing(trend="add", h=3, loss="tweedie").fit(y).predict())
    )
    assert np.all(np.isfinite(ExponentialSmoothing(h=3, loss="tweedie").fit(y).predict()))
    assert np.all(
        np.isfinite(ExponentialSmoothing(trend="damped", h=3, loss="tweedie").fit(y).predict())
    )
    y_seasonal = 20 + 0.5 * np.arange(40) + 5 * np.sin(2 * np.pi * np.arange(40) / 4)
    hw = ExponentialSmoothing(trend="add", h=3, seasonal_period=4, loss="tweedie").fit(y_seasonal)
    assert np.all(np.isfinite(hw.predict()))


def test_out_of_range_tweedie_p_raises():
    with pytest.raises(ValueError, match="p must be"):
        ExponentialSmoothing(trend="add", h=3, loss="tweedie", p=2.5)


def test_tweedie_negative_y_raises():
    y = np.array([1.0, 2.0, -1.0, 3.0, 4.0, 5.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="non-negative"):
        ExponentialSmoothing(trend="add", h=3, loss="tweedie").fit(y)


def test_tweedie_handles_zero_inflated_series_better_than_mse():
    # Intermittent-demand series: mostly zero, with small sporadic demand and
    # one huge one-off spike right before the forecast origin. MSE chases the
    # spike's squared error and extrapolates a large trend into the forecast;
    # tweedie's variance-scales-with-mean deviance is far less swayed by it.
    # Both fits must stay finite/reasonably scaled, but tweedie's forecast
    # error against a held-out, in-distribution (no repeat spike) future must
    # be measurably smaller than mse's.
    rng = np.random.default_rng(11)
    n, h = 30, 3
    y_fit = rng.gamma(2, 1, n) * rng.binomial(1, 0.3, n)
    y_fit[-1] = 500.0  # one huge, one-off demand spike right before the origin
    y_future = np.array([0.0, 3.0, 0.0])  # typical, in-distribution continuation

    mse_fit = ExponentialSmoothing(trend="add", h=h, loss="mse", n_starts=8).fit(y_fit)
    tweedie_fit = ExponentialSmoothing(trend="add", h=h, loss="tweedie", n_starts=8).fit(y_fit)

    mse_pred = mse_fit.predict()
    tweedie_pred = tweedie_fit.predict()
    assert np.all(np.isfinite(mse_pred))
    assert np.all(np.isfinite(tweedie_pred))
    # reasonably scaled: not orders of magnitude beyond the data's own range
    assert np.all(tweedie_pred < 10 * y_fit.max())

    err_mse = np.abs(mse_pred - y_future).mean()
    err_tweedie = np.abs(tweedie_pred - y_future).mean()
    assert err_tweedie < err_mse


# --- Missing-value support (#33: non-seasonal, #34: seasonal) ---


def test_states_np_missing_step_advances_by_trend():
    y = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
    alpha, beta = 0.5, 0.5
    levels, trends = holt_module._states_np(alpha, beta, 1.0, 0.0, 0.0, y)

    level, trend = 0.0, 0.0
    expected_levels, expected_trends = [level], [trend]
    for yt in y:
        if np.isnan(yt):
            new_level, new_trend = level + trend, trend
        else:
            new_level = alpha * yt + (1 - alpha) * (level + trend)
            new_trend = beta * (new_level - level) + (1 - beta) * trend
        level, trend = new_level, new_trend
        expected_levels.append(level)
        expected_trends.append(trend)

    np.testing.assert_allclose(levels, expected_levels)
    np.testing.assert_allclose(trends, expected_trends)


def test_holt_fits_and_predicts_with_interior_nan_gap():
    y = 3.0 + 2.0 * np.arange(30)
    y_gap = y.copy()
    y_gap[10:12] = np.nan
    forecasts = ExponentialSmoothing(trend="add", h=5, loss="mse").fit(y_gap).predict()
    expected = y[-1] + 2.0 * np.arange(1, 6)
    np.testing.assert_allclose(forecasts, expected, atol=0.5)


def test_ses_fits_with_interior_nan_gap():
    y = np.full(20, 7.0)
    y[10] = np.nan
    forecasts = ExponentialSmoothing(h=3, loss="mse").fit(y).predict()
    np.testing.assert_allclose(forecasts, np.full(3, 7.0), atol=0.5)


def test_damped_holt_fits_with_interior_nan_gap():
    y = np.linspace(0, 10, 20)
    y[10] = np.nan
    forecasts = ExponentialSmoothing(trend="damped", h=4, loss="mse").fit(y).predict()
    assert np.all(np.isfinite(forecasts))


def test_leading_nan_run_warns_once():
    y = 3.0 + 2.0 * np.arange(30)
    y[:3] = np.nan
    with pytest.warns(UserWarning, match="starts with a run of NaN") as record:
        ExponentialSmoothing(trend="add", h=5, loss="mse").fit(y)
    assert len(record) == 1


def test_trailing_nan_run_warns_once():
    y = 3.0 + 2.0 * np.arange(30)
    y[-2:] = np.nan
    with pytest.warns(UserWarning, match="ends with a run of NaN") as record:
        ExponentialSmoothing(trend="add", h=5, loss="mse").fit(y)
    assert len(record) == 1


@pytest.mark.parametrize("seasonal_period", [None, 4])
def test_trailing_nan_forecast_stays_on_trend(seasonal_period):
    # A missing step must still advance the state by one trend step; carrying the
    # level unchanged left the forecast one slope behind per trailing NaN.
    y = 3.0 + 2.0 * np.arange(40)
    y[-3:] = np.nan
    with pytest.warns(UserWarning, match="ends with a run of NaN"):
        model = ExponentialSmoothing(trend="add", seasonal_period=seasonal_period, h=5).fit(y)
    np.testing.assert_allclose(model.predict(), 3.0 + 2.0 * np.arange(40, 45), atol=0.05)


def test_leading_and_trailing_nan_runs_each_warn_once():
    y = 3.0 + 2.0 * np.arange(30)
    y[:2] = np.nan
    y[-2:] = np.nan
    with pytest.warns(UserWarning) as record:
        ExponentialSmoothing(trend="add", h=5, loss="mse").fit(y)
    assert len(record) == 2


def test_all_nan_raises():
    y = np.full(20, np.nan)
    with pytest.raises(ValueError, match="all NaN"):
        ExponentialSmoothing(trend="add", h=5, loss="mse").fit(y)


def test_holt_winters_all_nan_raises():
    y = np.full(20, np.nan)
    with pytest.raises(ValueError, match="all NaN"):
        ExponentialSmoothing(trend="add", h=3, seasonal_period=4).fit(y)


def test_predict_interval_correct_with_nan_in_fitted_series():
    y = 3.0 + 2.0 * np.arange(30)
    y[10] = np.nan
    model = ExponentialSmoothing(trend="add", h=5, loss="mse").fit(y)
    point = model.predict()
    lower, upper = model.predict_interval(coverage=0.9)
    assert np.all(np.isfinite(lower)) and np.all(np.isfinite(upper))
    assert np.all(lower <= point) and np.all(point <= upper)


def test_holt_objective_excludes_nan_actual_cells_from_every_origin_window():
    # Mirrors test_sample_weight_zero_excludes_observation_from_every_origin_window
    # (#25): a NaN actual must drop every (origin, k) cell it appears in, not
    # just the one where it's the origin.
    n_origins, h, target_index = 4, 3, 2
    actuals = np.arange(n_origins * h, dtype=float).reshape(n_origins, h)
    actuals_nan = actuals.copy()
    for t in range(n_origins):
        for k in range(h):
            if t + k == target_index:
                actuals_nan[t, k] = np.nan
    forecasts = np.zeros_like(actuals)

    actual_mask = holt_module._nan_actual_mask(actuals_nan)
    value = holt_module._loss_value_np(
        ("mse", 1.0, 0.5, 1.0, 1.5), actuals_nan, forecasts, actual_mask=actual_mask
    )

    kept = actuals[actual_mask]
    assert value == pytest.approx(float((kept**2).mean()))


def test_heuristic_init_skips_leading_nan():
    y = np.concatenate([[np.nan, np.nan], 3.0 + 2.0 * np.arange(15)])
    _level0, trend0 = holt_module._heuristic_init(y)
    assert trend0 == pytest.approx(2.0, abs=0.3)


def test_heuristic_init_degenerate_single_valid_point():
    y = np.array([np.nan, np.nan, 5.0, np.nan])
    level0, trend0 = holt_module._heuristic_init(y)
    assert level0 == pytest.approx(5.0)
    assert trend0 == pytest.approx(0.0)


def test_holt_winters_additive_fits_with_interior_nan_gap():
    m, h = 4, 4
    t = np.arange(44)
    seasonal_true = np.array([5.0, -3.0, -4.0, 2.0])
    y_full = 20.0 + 1.5 * t + seasonal_true[t % m]
    y_fit, y_future = y_full[:-h], y_full[-h:]
    y_gap = y_fit.copy()
    y_gap[10] = np.nan

    forecasts = (
        ExponentialSmoothing(trend="add", h=h, seasonal_period=m, loss="mse", n_starts=8)
        .fit(y_gap)
        .predict()
    )
    np.testing.assert_allclose(forecasts, y_future, atol=1.0)


def test_holt_winters_multiplicative_fits_with_interior_nan_gap():
    # Division-rule skip: the missing observation must be skipped entirely,
    # never divided by (#34).
    m, h = 4, 4
    t = np.arange(44)
    seasonal_true = np.array([1.2, 0.8, 0.9, 1.1])
    y_full = (20.0 + 1.5 * t) * seasonal_true[t % m]
    y_fit, y_future = y_full[:-h], y_full[-h:]
    y_gap = y_fit.copy()
    y_gap[10] = np.nan

    forecasts = (
        ExponentialSmoothing(
            trend="add",
            h=h,
            seasonal_period=m,
            seasonal_type="multiplicative",
            loss="mse",
            n_starts=8,
        )
        .fit(y_gap)
        .predict()
    )
    np.testing.assert_allclose(forecasts, y_future, atol=1.5)


def test_damped_holt_winters_fits_with_interior_nan_gap():
    m = 4
    t = np.arange(48)
    y = 100.0 + t.astype(float) + np.tile([5.0, -3.0, -4.0, 2.0], 12)
    y[20] = np.nan
    forecasts = (
        ExponentialSmoothing(trend="damped", h=4, seasonal_period=m, loss="mse", n_starts=5)
        .fit(y)
        .predict()
    )
    assert np.all(np.isfinite(forecasts))


def test_holt_winters_leading_nan_run_warns_once():
    m, h = 4, 4
    t = np.arange(44)
    seasonal_true = np.array([5.0, -3.0, -4.0, 2.0])
    y = 20.0 + 1.5 * t + seasonal_true[t % m]
    y[:2] = np.nan
    with pytest.warns(UserWarning, match="starts with a run of NaN") as record:
        ExponentialSmoothing(trend="add", h=h, seasonal_period=m, loss="mse").fit(y)
    assert len(record) == 1


def test_holt_winters_predict_interval_correct_with_nan_in_fitted_series():
    m, h = 4, 4
    t = np.arange(44)
    seasonal_true = np.array([5.0, -3.0, -4.0, 2.0])
    y = 20.0 + 1.5 * t + seasonal_true[t % m]
    y[10] = np.nan

    model = ExponentialSmoothing(trend="add", h=h, seasonal_period=m, loss="mse", n_starts=8).fit(y)
    point = model.predict()
    lower, upper = model.predict_interval(coverage=0.9)
    assert np.all(np.isfinite(lower)) and np.all(np.isfinite(upper))
    assert np.all(lower <= point) and np.all(point <= upper)


def test_heuristic_seasonal_init_skips_interior_nan():
    m = 4
    t = np.arange(12)
    seasonal_true = np.array([5.0, -3.0, -4.0, 2.0])
    y = 20.0 + 1.5 * t + seasonal_true[t % m]
    y[2] = np.nan

    level0, trend0, season0 = holt_module._heuristic_seasonal_init(y, m)
    assert np.isfinite(level0)
    assert np.isfinite(trend0)
    assert np.all(np.isfinite(season0))


def test_heuristic_seasonal_init_degenerate_single_valid_point():
    y = np.array([np.nan, np.nan, 5.0, np.nan, np.nan, np.nan, np.nan, np.nan])
    level0, trend0, season0 = holt_module._heuristic_seasonal_init(y, m=4)
    assert level0 == pytest.approx(5.0)
    assert trend0 == pytest.approx(0.0)
    np.testing.assert_allclose(season0, np.zeros(4))


def test_heuristic_seasonal_init_multiplicative_degenerate_single_valid_point():
    y = np.array([np.nan, np.nan, 5.0, np.nan, np.nan, np.nan, np.nan, np.nan])
    level0, trend0, season0 = holt_module._heuristic_seasonal_init_multiplicative(y, m=4)
    assert level0 == pytest.approx(5.0)
    assert trend0 == pytest.approx(0.0)
    np.testing.assert_allclose(season0, np.ones(4))
