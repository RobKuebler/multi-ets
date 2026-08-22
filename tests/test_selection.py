import numpy as np

from multi_ets import SES, Holt, HoltWinters, select_model


def test_select_model_picks_seasonal_model_on_seasonal_data():
    m, h = 4, 4
    t = np.arange(60)
    seasonal_true = np.array([5.0, -3.0, -4.0, 2.0])
    y = 20.0 + 1.5 * t + seasonal_true[t % m]

    candidates = [
        SES(h=h, loss="mae"),
        HoltWinters(h=h, seasonal_period=m, loss="mae", n_starts=8),
    ]
    best, scores = select_model(candidates, y, n_splits=3)

    assert isinstance(best, HoltWinters)
    assert len(scores) == len(candidates)
    for candidate in candidates:
        assert candidate in scores


def test_select_model_refits_best_on_full_series():
    y = 3.0 + 2.0 * np.arange(40)
    candidates = [SES(h=3, loss="mse")]
    best, _ = select_model(candidates, y, n_splits=2)
    assert np.allclose(best._y, y)


def test_select_model_scores_custom_callable_loss_candidate():
    # select_model must score a callable loss the same way SES/Holt.fit()
    # itself does: by calling it directly, not via the (now-deleted)
    # jax-based _loss_fn_from_key.
    def weighted_mse(y_true, y_pred):
        return np.mean((y_pred - y_true) ** 2)

    y = 3.0 + 2.0 * np.arange(30)
    candidates = [SES(h=3, loss=weighted_mse)]
    best, scores = select_model(candidates, y, n_splits=2)
    assert isinstance(best, SES)
    assert scores[candidates[0]] >= 0.0


def test_select_model_ignores_nan_actual_in_a_cv_test_fold():
    # A NaN landing in a CV test fold must be zero-weighted in that fold's
    # score (mirrors holt.py's own _nan_actual_mask), not poison the whole
    # candidate's mean CV loss to NaN.
    y = 3.0 + 2.0 * np.arange(30)
    y_gap = y.copy()
    y_gap[-1] = np.nan  # lands in the last split's test fold

    candidates = [Holt(h=3, loss="mse")]
    _, scores = select_model(candidates, y_gap, n_splits=2)
    assert np.isfinite(scores[candidates[0]])
