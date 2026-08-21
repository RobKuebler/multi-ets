import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from statsmodels.tsa.holtwinters import Holt as SMHolt

from multi_ets import SES, DampedHolt, DampedHoltWinters, Holt, HoltWinters
from multi_ets import holt as holt_module


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


@pytest.mark.parametrize(
    "loss_name,delta,tau",
    [("mse", 1.0, 0.5), ("mae", 1.0, 0.5), ("huber", 0.7, 0.5), ("pinball", 1.0, 0.3)],
)
@pytest.mark.parametrize("kind,n_params", [("undamped", 4), ("damped", 5), ("ses", 2)])
def test_hand_derived_grad_matches_jax_autodiff(kind, n_params, loss_name, delta, tau):
    # #18: Holt.fit()'s hand-derived BPTT gradient (_holt_objective_and_grad)
    # must agree with the JAX autodiff gradient it replaces, for every
    # built-in loss and every non-seasonal kind (undamped/damped/ses).
    rng = np.random.default_rng(0)
    h = 3
    y = np.cumsum(rng.normal(0, 1, 20)) + 10
    unpack = holt_module._UNPACK_BY_KIND[kind]
    loss_key = (loss_name, delta, tau)
    loss_fn_jax = holt_module._loss_fn_from_key(loss_key)

    p = rng.uniform(0.1, 0.9, n_params)
    if kind == "damped":
        p[2] = min(p[2], 0.9)  # phi < 1, matching the classical damping range

    def jax_loss(params):
        return holt_module._joint_loss(params, jnp.asarray(y), h, loss_fn_jax, unpack)

    jax_grad = np.asarray(jax.grad(jax_loss)(jnp.asarray(p)))
    _, grad = holt_module._holt_objective_and_grad(p, y, h, kind, loss_key)

    np.testing.assert_allclose(grad, jax_grad, atol=1e-6)


def test_hand_derived_grad_matches_jax_autodiff_with_padding():
    # Holt.fit() always fits on a bucket-padded, masked series (_pad_to_bucket)
    # -- the branch above never exercises, since it never passes n_valid. That
    # masked branch is the one every real .fit() call actually uses, so it
    # needs its own jax-vs-numpy cross-check.
    rng = np.random.default_rng(1)
    h = 3
    y = np.cumsum(rng.normal(0, 1, 13)) + 10  # 13 -> pads to 16
    y_padded = holt_module._pad_to_bucket(y)
    assert len(y_padded) > len(y)  # sanity: this length actually needs padding
    kind = "undamped"
    unpack = holt_module._UNPACK_BY_KIND[kind]
    loss_key = ("mse", 1.0, 0.5)
    loss_fn_jax = holt_module._loss_fn_from_key(loss_key)
    p = rng.uniform(0.1, 0.9, 4)

    def jax_loss(params):
        return holt_module._joint_loss(
            params, jnp.asarray(y_padded), h, loss_fn_jax, unpack, n_valid=len(y)
        )

    jax_grad = np.asarray(jax.grad(jax_loss)(jnp.asarray(p)))
    _, grad = holt_module._holt_objective_and_grad(p, y_padded, h, kind, loss_key, n_valid=len(y))

    np.testing.assert_allclose(grad, jax_grad, atol=1e-6)


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


def test_damped_phi_capped_on_strongly_trending_series():
    # Superlinear growth pins phi against the upper bound; it must stop at
    # the standard classical damping cap (0.98), not collapse to undamped (~1).
    y = 100.0 + np.arange(40, dtype=float) ** 2
    model = DampedHolt(h=4, loss="mse", n_starts=10).fit(y)
    phi = model.params_[2]
    assert phi <= 0.98


def test_damped_holt_winters_phi_capped_on_strongly_trending_series():
    m = 4
    t = np.arange(48)
    y = 100.0 + t.astype(float) ** 2 + np.tile([5.0, -3.0, -4.0, 2.0], 12)
    model = DampedHoltWinters(h=4, seasonal_period=m, loss="mse", n_starts=10).fit(y)
    phi = model.params_[2]
    assert phi <= 0.98


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


def test_compiled_multistart_cache_stays_bounded_across_many_lengths(monkeypatch):
    # Regression test for #15: fitting many distinct training lengths (as a
    # rolling-origin backtest does) must not grow the compiled-closure cache
    # without bound -- each length beyond the cache's maxsize should evict an
    # older one rather than accumulate forever. Swaps in a maxsize=2 cache
    # (same underlying builder) so the eviction behavior can be proven with a
    # handful of real compiles instead of _COMPILE_CACHE_SIZE + 1 of them.
    #
    # Calls _compiled_multistart directly rather than through SES.fit(): as of
    # #18, nothing in this module calls this JAX-jaxopt cache any more (it's
    # fully dead code, left in place until #20 deletes it); this test still
    # exercises the cache's own eviction behavior, which is otherwise untested.
    def _build(h, loss_key, kind, n):  # mirrors _compiled_multistart_cached's signature
        return holt_module._compile_multistart(h, loss_key, kind)

    small_cache = functools.lru_cache(maxsize=2)(_build)
    monkeypatch.setattr(holt_module, "_compiled_multistart_cached", small_cache)

    loss_key = holt_module._loss_key("mse", 1.0, 0.5)
    for length in range(5, 9):  # 4 distinct lengths through a maxsize=2 cache
        holt_module._compiled_multistart(1, loss_key, "ses", length)

    assert small_cache.cache_info().currsize <= 2


def test_bucket_length_is_smallest_power_of_two_at_least_n():
    assert holt_module._bucket_length(1) == 1
    assert holt_module._bucket_length(5) == 8
    assert holt_module._bucket_length(8) == 8
    assert holt_module._bucket_length(9) == 16


def test_pad_to_bucket_repeats_last_value_and_is_a_noop_at_a_power_of_two():
    padded = holt_module._pad_to_bucket(np.array([1.0, 2.0, 3.0]))
    np.testing.assert_array_equal(padded, [1.0, 2.0, 3.0, 3.0])

    y8 = np.arange(8.0)
    assert holt_module._pad_to_bucket(y8) is y8  # already a bucket length: no-op


def test_padded_joint_loss_is_bit_exact_with_unpadded():
    # Correctness bar for #17: padding a training array to a bucket length
    # and masking out the padded origins must reproduce the unpadded joint
    # loss bit-for-bit, not just approximately -- fitted params must not move.
    y = np.linspace(-1.0, 2.0, 13)  # 13 -> pads to 16
    h = 3
    unpack = holt_module._UNPACK_BY_KIND["undamped"]
    loss_fn = holt_module._loss_fn_from_key(holt_module._loss_key("mse", 1.0, 0.5))
    params = np.array([0.3, 0.2, 5.0, 1.0])

    unpadded = holt_module._joint_loss(params, jnp.asarray(y), h, loss_fn, unpack)

    y_padded = holt_module._pad_to_bucket(y)
    assert len(y_padded) > len(y)  # sanity: this length actually needs padding
    padded = holt_module._joint_loss(
        params, jnp.asarray(y_padded), h, loss_fn, unpack, n_valid=len(y)
    )

    assert padded == unpadded


def test_expanding_window_refits_share_one_compiled_shape(monkeypatch):
    # Regression test for #17: rolling-origin backtests refit at every
    # expanding-window length (n, n+h, n+2h, ...), and each distinct length
    # used to force a fresh XLA compile since a fresh length never repeats
    # within one backtest. Bucketing to the next power of two means every
    # length in [5, 8] now shares bucket 8, so this cache should see the same
    # single entry hit on every refit after the first.
    #
    # Simulates the bucketing + cache lookup directly rather than through
    # SES.fit(): as of #18, SES/Holt/DampedHolt no longer pass a bucketed
    # length into this JAX-jaxopt cache themselves (it's dead code from their
    # side now, kept alive only until #20 deletes it); this test still
    # exercises the cache's own bucket-sharing behavior, which is otherwise
    # untested.
    def _build(h, loss_key, kind, n):  # mirrors _compiled_multistart_cached's signature
        return holt_module._compile_multistart(h, loss_key, kind)

    small_cache = functools.lru_cache(maxsize=2)(_build)
    monkeypatch.setattr(holt_module, "_compiled_multistart_cached", small_cache)

    loss_key = holt_module._loss_key("mse", 1.0, 0.5)
    for length in range(5, 9):  # expanding window: 5, 6, 7, 8
        bucket_len = len(holt_module._pad_to_bucket(np.linspace(0, 1, length)))
        holt_module._compiled_multistart(1, loss_key, "ses", bucket_len)

    assert small_cache.cache_info().hits >= 3  # only the first length misses
    assert small_cache.cache_info().currsize == 1


def test_multiplicative_seasonal_fit_skips_bucket_padding(monkeypatch):
    # Regression test: the multiplicative seasonal recursion (y_t/s_old,
    # y_t/level) is a ratio, not a convex combination like every other kind,
    # so it isn't provably bounded the way the additive/non-seasonal kinds
    # are (spectral radius < 1 everywhere in the alpha/beta/(phi) box).
    # Direct simulation confirms it blows up toward +-inf/0 well within
    # float64 range at plausible multi-start corners when iterated hundreds
    # of steps beyond the real series length -- exactly what bucket-padding
    # would do -- so #17's padding must never apply here.
    def _boom(y):
        raise AssertionError("must not bucket-pad a multiplicative-seasonal fit")

    monkeypatch.setattr(holt_module, "_pad_to_bucket", _boom)
    y = 10.0 + 0.1 * np.arange(30) + np.tile([1.0, 1.2, 0.8], 10)
    model = HoltWinters(h=2, seasonal_period=3, seasonal_type="multiplicative", n_starts=2).fit(y)
    assert np.all(np.isfinite(model.params_))
