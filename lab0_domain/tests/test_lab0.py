"""Tests of the shared data layer (dmfin) used by every laboratory.

All tests run on synthetic data and need no network access.
"""
import numpy as np
import pandas as pd
import pytest

from dmfin import features as F
from dmfin import plotting as P
from dmfin.reporting import _macro_name
from dmfin.splits import chrono_split, expanding_window_cv, make_windows


@pytest.fixture
def prices():
    idx = pd.bdate_range("2010-01-01", periods=300)
    rng = np.random.default_rng(0)
    r = rng.normal(0, 0.01, len(idx))
    close = 100 * np.exp(np.cumsum(r))
    high = close * np.exp(np.abs(rng.normal(0, 0.005, len(idx))))
    low = close * np.exp(-np.abs(rng.normal(0, 0.005, len(idx))))
    return pd.DataFrame({"close": close, "high": high, "low": low}, index=idx)


def test_parkinson_constant_range():
    # ln(H/L) = 0.02 every day -> variance = 0.02^2 / (4 ln 2)
    h = pd.Series(np.full(5, np.exp(0.02)))
    low = pd.Series(np.ones(5))
    v = F.parkinson_var(h, low)
    assert np.allclose(v, 0.02 ** 2 / (4 * np.log(2)))


def test_future_target_uses_only_future(prices):
    r = F.log_returns(prices["close"])
    y = F.future_realized_vol(r, 5)
    t = 100
    manual = np.sqrt(252 / 5 * (r.iloc[t + 1:t + 6] ** 2).sum())
    assert y.iloc[t] == pytest.approx(manual)
    # changing the past must not change the target
    r2 = r.copy(); r2.iloc[: t + 1] = 0.5
    assert F.future_realized_vol(r2, 5).iloc[t] == pytest.approx(manual)


def test_past_feature_uses_only_past(prices):
    r = F.log_returns(prices["close"])
    x = F.past_realized_vol(r, 22)
    r2 = r.copy(); r2.iloc[151:] = 1.0  # change the future
    assert F.past_realized_vol(r2, 22).iloc[150] == pytest.approx(x.iloc[150])


def test_chrono_split_is_ordered_and_purged():
    idx = pd.bdate_range("2013-01-01", "2020-12-31")
    sp = chrono_split(idx, horizon=5, train_end="2014-12-31", val_end="2016-12-31",
                      test_end="2020-12-31")
    assert sp.train.max() < sp.val.min() < sp.val.max() < sp.test.min()
    # the last training sample's 5-day target window must end inside training
    last_pos = idx.get_loc(sp.train.max())
    assert idx[last_pos + 5] <= pd.Timestamp("2014-12-31")
    assert len(set(sp.train) & set(sp.val)) == 0


def test_expanding_cv_embargo():
    n, gap = 1000, 22
    for tr, va in expanding_window_cv(n, n_splits=4, gap=gap):
        assert tr.max() + gap < va.min()  # embargo respected
        assert va.min() >= 400


def test_make_windows_shapes_and_alignment():
    v = np.arange(50, dtype=float).reshape(-1, 1)
    y = np.arange(50, dtype=float) * 10
    X, yy, pos = make_windows(v, y, np.arange(0, 50), lookback=7)
    assert X.shape == (44, 7, 1)
    assert pos[0] == 6 and X[0, -1, 0] == 6 and yy[0] == 60


def test_macro_names_are_latex_safe():
    assert _macro_name("lab2", "test_rmse_2") == "LabTwoTestRmseTwo"


# ------------------------------------------------- extended data (iteration 2)
def test_ken_french_table_parser_converts_percent_and_missing():
    from dmfin.data_ext import _ff_table
    txt = ("Header line\n\n  Average Value Weighted Returns -- Daily\n,Agric,Food\n"
           "20000103,   1.00, -99.99\n20000104,  -0.50,   0.25\n\n  Average Equal Weighted Returns -- Daily\n"
           ",Agric,Food\n20000103,   9.00,   9.00\n")
    df = _ff_table(txt, title="Average Value Weighted Returns -- Daily")
    assert list(df.columns) == ["Agric", "Food"] and len(df) == 2
    assert df.iloc[0, 0] == pytest.approx(0.01) and np.isnan(df.iloc[0, 1])


def test_fredmd_transformation_codes():
    from dmfin.data_ext import transform_fredmd
    lv = pd.DataFrame({"a": [1.0, 2.0, 4.0, 8.0], "b": [1.0, 2.0, 4.0, 8.0], "c": [1.0, 3.0, 6.0, 10.0]})
    out = transform_fredmd(lv, pd.Series({"a": 1, "b": 5, "c": 2}))
    assert out["a"].tolist() == lv["a"].tolist()
    assert out["b"].iloc[1:].tolist() == pytest.approx([np.log(2)] * 3)       # growth rate of a doubling series
    assert out["c"].iloc[1:].tolist() == [2.0, 3.0, 4.0]


def test_sic_codes_map_to_fama_french_industries():
    from dmfin.data_ext import sic_to_ff49
    table = pd.DataFrame({"no": [1, 2], "short": ["Agric", "Food"], "name": ["A", "F"], "lo": [100, 2000], "hi": [199, 2046]})
    assert sic_to_ff49(150, table) == "Agric" and sic_to_ff49(2010, table) == "Food"
    assert sic_to_ff49(9999, table) == "Other" and sic_to_ff49(None, table) is None


def test_published_series_are_used_only_after_their_publication():
    from dmfin.data import available_at
    weekly = pd.Series([1.0, 2.0], index=pd.to_datetime(["2020-01-03", "2020-01-10"]))     # Fridays
    days = pd.bdate_range("2020-01-06", "2020-01-17")
    a = available_at(weekly, days, lag_days=7)
    assert a.loc[:"2020-01-09"].isna().all()                                        # nothing published yet
    assert (a.loc["2020-01-10":"2020-01-16"] == 1.0).all() and a.loc["2020-01-17"] == 2.0  # released a week later
    daily = pd.Series(np.arange(5.0), index=pd.bdate_range("2020-01-06", periods=5))
    b = available_at(daily, daily.index, lag_days=1)
    assert np.isnan(b.iloc[0]) and (b.iloc[1:].to_numpy() == daily.iloc[:-1].to_numpy()).all()


def test_per_cent_of_a_heatmap_cell():
    assert P.per_cent(0.613) == "61"
    assert P.per_cent(-0.613) == "\u221261"
    assert P.per_cent(1) == "100"
    assert P.per_cent(-0.002) == "0"            # no minus zero


def test_label_points_do_not_overlap():
    f, ax = P.fig(2.0, w=3.0)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    P.label_points(ax, [0.5] * 4, [0.5] * 4, ["AAAA", "BBBB", "CCCC", "DDDD"])
    r = f.canvas.get_renderer()
    boxes = [t.get_window_extent(r) for t in ax.texts]
    P.plt.close(f)
    assert len(boxes) == 4
    assert not any(a.overlaps(b) for i, a in enumerate(boxes) for b in boxes[i + 1:])
