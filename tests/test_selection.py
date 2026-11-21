import numpy as np
import pytest

from multi_ets import AutoExponentialSmoothing, ExponentialSmoothing, select_model


def test_select_model_picks_seasonal_model_on_seasonal_data():
    m, h = 4, 4
    t = np.arange(60)
    seasonal_true = np.array([5.0, -3.0, -4.0, 2.0])
    y = 20.0 + 1.5 * t + seasonal_true[t % m]

    candidates = [
        ExponentialSmoothing(h=h, loss="mae"),
        ExponentialSmoothing(trend="add", h=h, seasonal_period=m, loss="mae", n_starts=8),
    ]
    best, scores = select_model(candidates, y, n_splits=3)

    assert best.seasonal_period is not None
    assert len(scores) == len(candidates)
    for candidate in candidates:
        assert candidate in scores


def test_select_model_refits_best_on_full_series():
    y = 3.0 + 2.0 * np.arange(40)
    candidates = [ExponentialSmoothing(h=3, loss="mse")]
    best, _ = select_model(candidates, y, n_splits=2)
    assert np.allclose(best._y, y)


def test_select_model_scores_custom_callable_loss_candidate():
    # select_model must score a callable loss the same way ExponentialSmoothing.fit()
    # itself does: by calling it directly, not via the (now-deleted)
    # jax-based _loss_fn_from_key.
    def weighted_mse(y_true, y_pred):
        return np.mean((y_pred - y_true) ** 2)

    y = 3.0 + 2.0 * np.arange(30)
    candidates = [ExponentialSmoothing(h=3, loss=weighted_mse)]
    best, scores = select_model(candidates, y, n_splits=2)
    assert best.trend is None and best.seasonal_period is None
    assert scores[candidates[0]] >= 0.0


def test_select_model_ignores_nan_actual_in_a_cv_test_fold():
    # A NaN landing in a CV test fold must be zero-weighted in that fold's
    # score (mirrors holt.py's own _nan_actual_mask), not poison the whole
    # candidate's mean CV loss to NaN.
    y = 3.0 + 2.0 * np.arange(30)
    y_gap = y.copy()
    y_gap[-1] = np.nan  # lands in the last split's test fold

    candidates = [ExponentialSmoothing(trend="add", h=3, loss="mse")]
    _, scores = select_model(candidates, y_gap, n_splits=2)
    assert np.isfinite(scores[candidates[0]])


def test_auto_exponential_smoothing_without_seasonal_period_selects_among_trends():
    y = 3.0 + 2.0 * np.arange(40)
    model = AutoExponentialSmoothing(h=3, n_splits=2, n_starts=5).fit(y)

    assert isinstance(model.best_, ExponentialSmoothing)
    assert model.best_.seasonal_period is None
    assert len(model.scores_) == 3
    assert {c.trend for c in model.scores_} == {None, "add", "damped"}


def test_auto_exponential_smoothing_with_seasonal_period_selects_full_grid():
    m, h = 4, 4
    t = np.arange(60)
    seasonal_true = np.array([5.0, -3.0, -4.0, 2.0])
    y = 20.0 + 1.5 * t + seasonal_true[t % m]

    model = AutoExponentialSmoothing(h=h, seasonal_period=m, n_splits=2, n_starts=5).fit(y)

    assert len(model.scores_) == 6  # 3 trends x 2 seasonal_types
    configs = {(c.trend, c.seasonal_type) for c in model.scores_}
    assert configs == {
        (None, "additive"),
        (None, "multiplicative"),
        ("add", "additive"),
        ("add", "multiplicative"),
        ("damped", "additive"),
        ("damped", "multiplicative"),
    }
    # the no-trend-seasonal configs from #38 must be reachable candidates
    assert any(c.trend is None and c.seasonal_period is not None for c in model.scores_)


def test_auto_exponential_smoothing_best_is_refit_exponential_smoothing_instance():
    y = 3.0 + 2.0 * np.arange(40)
    model = AutoExponentialSmoothing(h=3, n_splits=2, n_starts=5).fit(y)
    assert type(model.best_) is ExponentialSmoothing
    assert np.allclose(model.best_._y, y)


def test_auto_exponential_smoothing_predict_and_predict_interval_delegate():
    y = 3.0 + 2.0 * np.arange(40)
    model = AutoExponentialSmoothing(h=3, n_splits=2, n_starts=5).fit(y)

    np.testing.assert_allclose(model.predict(), model.best_.predict())
    lower, upper = model.predict_interval(coverage=0.9)
    best_lower, best_upper = model.best_.predict_interval(coverage=0.9)
    np.testing.assert_allclose(lower, best_lower)
    np.testing.assert_allclose(upper, best_upper)


def test_auto_exponential_smoothing_predict_before_fit_raises():
    model = AutoExponentialSmoothing(h=3)
    with pytest.raises(ValueError, match="not fitted"):
        model.predict()


def test_select_model_with_list_h_scores_listed_steps():
    y = 3.0 + 2.0 * np.arange(40)
    candidates = [ExponentialSmoothing(h=[1, 4]), ExponentialSmoothing(trend="add", h=[1, 4])]
    best, scores = select_model(candidates, y, n_splits=2)
    assert best.trend == "add"
    assert best.predict().shape == (2,)
    assert len(scores) == 2


def test_auto_exponential_smoothing_with_list_h():
    y = 3.0 + 2.0 * np.arange(40)
    model = AutoExponentialSmoothing(h=[2, 5], n_starts=2, n_splits=2).fit(y)
    np.testing.assert_allclose(model.predict(), y[-1] + 2.0 * np.array([2, 5]), atol=1e-2)
