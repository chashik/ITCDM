"""Lab 10 tests (synthetic data, offline): gaps, segmentation, same-unit table, fuzzy tools, download lists."""
import numpy as np
import pandas as pd
import pytest

from lab10_economies.collect.download import FILES, belstat_files, minfin_files
from lab10_economies.fuzzy_tools import QUANT, best_quantifier, ruspini, summaries, tsk_design
from lab10_economies.growth import bond_return, same_units
from lab10_economies.prepare import benchmark
from lab10_economies.patterns import segments
from lab10_economies.prepare import fill_short_gaps


def test_short_gaps_are_interpolated_and_long_gaps_kept():
    s = pd.Series([1.0, np.nan, np.nan, 4.0, np.nan, np.nan, np.nan, 8.0, np.nan])
    f = fill_short_gaps(s)
    assert f.iloc[1:3].tolist() == pytest.approx([2.0, 3.0])        # two missing months: filled
    assert f.iloc[4:7].isna().all() and np.isnan(f.iloc[8])        # three months, and a gap at the end: kept


def test_exact_segmentation_finds_the_true_change_points():
    rng = np.random.default_rng(0)
    y = np.r_[np.full(20, 2.0), np.full(15, 12.0), np.full(25, 5.0)] + rng.normal(0, 0.3, 60)
    assert segments(y) == [20, 35]


def test_same_units_table_of_a_steady_economy():
    months = pd.period_range("2020-12", "2025-12", freq="M")
    g = 1.10 ** (1 / 12)                                           # everything grows 10 % a year
    p = pd.DataFrame({f"price_{k}": 100 * g ** np.arange(len(months)) for k in ("flour_wheat", "milk", "sugar", "petrol")},
                     index=months)
    p["cpi_index"] = 100 * g ** np.arange(len(months))
    p["cpi_yoy"], p["policy_rate"], p["bond_gov"], p["bond_market"] = 10.0, 12.0, 11.0, 11.0
    p["fx_usd"] = 2.0 * 1.05 ** (np.arange(len(months)) / 12)     # the dollar rises 5 % a year
    t = same_units(p)                                              # a yield that does not change is the holder's return
    assert t.loc["2023"].tolist() == pytest.approx([10, 10, 10, 10, 10, 12, 11, 11])
    assert t.loc["2021-2025, per year"].tolist() == pytest.approx([10, 10, 10, 10, 10, 12, 11, 11])
    usd = same_units(p, "BY")                                      # dollar bonds: 11 % in dollars, in roubles 1.11 * 1.05
    assert usd.loc["2023", "government bond return"] == pytest.approx(16.55)


def test_benchmark_matches_the_published_december_rates():
    months = [str(m) for m in pd.period_range("2020-12", "2022-12", freq="M")]
    chained = pd.Series(100 * 1.0081 ** np.arange(len(months)), index=months)  # rounded monthly changes: 10.2 % a year
    yoy = pd.Series({"2021-12": 10.0, "2022-12": 12.0})
    b = benchmark(chained, yoy)
    assert b["2021-12"] == pytest.approx(110.0) and b["2022-12"] == pytest.approx(110.0 * 1.12)
    assert b["2020-12"] == 100 and b["2021-06"] < b["2021-07"]                # the base and the monthly path are kept


def test_bond_return_falls_when_the_yield_rises():
    y = pd.Series([5.0, 5.0, 6.0, 6.0], index=pd.period_range("2021-01", periods=4, freq="M"))
    r = bond_return(y)                                             # 10 years, coupon 5 %, priced at 6 %, a month later
    price = 0.05 * (1 - 1.06 ** -10) / 0.06 + 1.06 ** -10
    assert r.iloc[2] == pytest.approx(100 * (1.06 ** (1 / 12) * price - 1)) and r.iloc[2] < -6
    assert r.iloc[1] == pytest.approx(100 * (1.05 ** (1 / 12) - 1))            # unchanged yield: one month of interest
    assert bond_return(pd.Series([0.0, 0.0]), 10).iloc[1] == pytest.approx(0)   # zero yield: no division by zero


def test_ruspini_partition_adds_up_to_one():
    g = np.linspace(-20, 20, 401)
    mu = ruspini(g, 2, 6)
    assert np.allclose(sum(mu.values()), 1)
    assert mu["about equal"][200] == 1 and mu["slower"][0] == 1 and mu["faster"][-1] == 1


def test_linguistic_summary_truth_and_covering():
    s = np.r_[np.ones(9), np.zeros(1)]                             # S holds fully in 9 of 10 records
    rows = summaries(pd.DataFrame(index=range(10)), {"S": s}, {"(all)": None}, "XX", "p", {"S": 0.3})
    best = best_quantifier(pd.DataFrame(rows)).iloc[0]
    assert best["proportion r"] == pytest.approx(0.9) and best["quantifier Q"] == "most"   # "almost all": 0.67
    assert best["T1 truth"] == 1 and float(QUANT["almost all"](0.9)) == pytest.approx(2 / 3)
    assert best["T3 covering"] == pytest.approx(0.9) and best["T2 imprecision (high = precise S)"] == pytest.approx(0.7)
    w = np.r_[np.zeros(5), np.ones(5)]                             # qualifier: only the last five records
    q = pd.DataFrame(summaries(pd.DataFrame(index=range(10)), {"S": s}, {"W": w}, "XX", "p", {"S": 0.3}))
    assert q["proportion r"].iloc[0] == pytest.approx(0.8)         # 4 of the 5 records with W


def test_takagi_sugeno_firing_strengths_are_normalized():
    X, names = tsk_design(np.array([1.0, 8.0, 20.0]), np.array([-2.0, 0.0, 2.0]))
    assert X.shape == (3, 6) and len(names) == 6 and np.allclose(X.sum(1), 1)
    X1, _ = tsk_design(np.array([8.0]), np.array([0.0]), order=1)
    assert X1.shape == (1, 12) and X1[0, 6:].sum() == pytest.approx(8.0)


def test_download_lists():
    b, m = belstat_files(), minfin_files()
    assert len(b) == 61 and "belstat/goods/goods_2020-12.xlsx" in b
    assert len(m) == 61 and "minfin_by/register/obl_obr_0126.xlsx" in m
    assert all(u.startswith("https://") for u in {**FILES, **b, **m}.values())
