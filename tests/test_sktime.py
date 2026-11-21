import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sktime")

from sktime.forecasting.conformal import ConformalIntervals
from sktime.utils.estimator_checks import check_estimator

from multi_ets import ExponentialSmoothing
from multi_ets.integrations.sktime import AutoMultiETSForecaster, MultiETSForecaster


def _y(n=60, index=None):
    rng = np.random.default_rng(0)
    values = 10 + 0.5 * np.arange(n) + rng.normal(0, 1, n)
    return pd.Series(values, index=index if index is not None else pd.RangeIndex(n), name="y")


@pytest.mark.filterwarnings("ignore:fewer than 30 in-sample residuals")
@pytest.mark.parametrize("cls", [MultiETSForecaster, AutoMultiETSForecaster])
def test_check_estimator(cls):
    check_estimator(cls, raise_exceptions=True)


def test_non_contiguous_fh_matches_core():
    y, steps = _y(), [1, 2, 3, 7, 8, 9]
    pred = MultiETSForecaster(trend="add", n_starts=2).fit(y, fh=steps).predict()
    core = ExponentialSmoothing(h=steps, trend="add", n_starts=2).fit(y.to_numpy()).predict()
    np.testing.assert_allclose(pred.to_numpy(), core)
    assert list(pred.index) == [59 + s for s in steps]
    assert pred.name == "y"


def test_mixed_insample_and_out_of_sample_fh():
    y = _y()
    f = MultiETSForecaster(trend="add", n_starts=2, insample_step=2).fit(y, fh=[-2, -1, 0, 1, 2])
    core = ExponentialSmoothing(h=[1, 2], trend="add", n_starts=2).fit(y.to_numpy())
    pred = f.predict()
    np.testing.assert_allclose(pred.to_numpy()[:3], core.predict_in_sample(step=2)[-3:])
    np.testing.assert_allclose(pred.to_numpy()[3:], core.predict())


def test_nan_in_y():
    y = _y()
    y.iloc[[10, 30]] = np.nan
    pred = MultiETSForecaster(trend="add", n_starts=2).fit(y, fh=[1, 2]).predict()
    assert np.isfinite(pred).all()


@pytest.mark.parametrize(
    "index",
    [
        pd.period_range("2020-01", periods=60, freq="M"),
        pd.date_range("2020-01-01", periods=60, freq="D"),
    ],
)
def test_period_and_datetime_index(index):
    y = _y(index=index)
    pred = MultiETSForecaster(n_starts=2).fit(y, fh=[1, 2, 3]).predict()
    pd.testing.assert_index_equal(pred.index, index.shift(3)[-3:])


def test_predict_interval_format():
    y = _y()
    f = MultiETSForecaster(trend="add", n_starts=2).fit(y, fh=[1, 2, 3])
    interval = f.predict_interval(coverage=[0.5, 0.9])
    assert list(interval.columns) == [("y", c, b) for c in (0.5, 0.9) for b in ("lower", "upper")]
    assert (interval[("y", 0.9, "lower")] <= interval[("y", 0.5, "lower")]).all()


def test_conformal_intervals_wrapping():
    y = _y(index=pd.period_range("2020-01", periods=60, freq="M"))  # RangeIndex fh reads relative
    f = ConformalIntervals(MultiETSForecaster(n_starts=1), initial_window=40)
    interval = f.fit(y, fh=[1, 2]).predict_interval(coverage=0.9)
    assert interval.shape == (2, 2)
    assert interval.notna().all().all()


def test_multivariate_y_is_vectorized():
    y = pd.DataFrame({"a": _y().to_numpy(), "b": _y().to_numpy()[::-1]})
    pred = AutoMultiETSForecaster(n_starts=1, n_splits=2).fit(y, fh=[1, 2]).predict()
    assert list(pred.columns) == ["a", "b"]
    assert pred.shape == (2, 2)
