import numpy as np

from multi_ets import SES, HoltWinters, select_model


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
