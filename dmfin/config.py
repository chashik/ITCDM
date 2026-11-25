"""Project-wide constants: paths, sample period, chronological split, seeds.

Every laboratory imports its dates from here, so that the training,
validation and test periods are identical across all experiments and no
script can silently use a different split.
"""
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent.parent
PROJECT_DIR = CODE_DIR.parent
DATA_DIR = CODE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROC_DIR = DATA_DIR / "processed"
RESULTS_DIR = CODE_DIR / "results"
FIG_DIR = PROJECT_DIR / "paper" / "figures"
GEN_DIR = PROJECT_DIR / "paper" / "generated"

# Sample period of the market panel (fixed so that every run is reproducible).
START = "2000-01-01"
END = "2026-06-30"

# Chronological split used by all time-series laboratories.
#   train      : 2000-01-03 .. 2014-12-31  (dot-com bust, 2008 crisis, recovery)
#   validation : 2015-01-02 .. 2018-12-31  (model and hyperparameter selection)
#   test       : 2019-01-02 .. 2026-06-30  (COVID-19 crash, 2022 bear market, ...)
TRAIN_END = "2014-12-31"
VAL_END = "2018-12-31"
TEST_END = END

# Cross-sectional stock universe (Lab 1): ten years of daily prices.
STOCKS_START = "2016-07-01"
STOCKS_SPLIT = "2023-06-30"  # clusters are fitted before, re-checked after

SEED = 2026
TRADING_DAYS = 252

# FRED series (daily unless noted). Values of day t are end-of-day values,
# i.e. they are known after the close of day t.
FRED_SERIES = {
    "VIXCLS": "CBOE Volatility Index (VIX), close",
    "DGS10": "10-year Treasury constant-maturity yield, %",
    "DGS2": "2-year Treasury constant-maturity yield, %",
    "DTB3": "3-month Treasury bill secondary-market rate, %",
    "DBAA": "Moody's seasoned Baa corporate bond yield, %",
    "DAAA": "Moody's seasoned Aaa corporate bond yield, %",
    "DFF": "Effective federal funds rate, %",
    "DCOILWTICO": "WTI crude oil spot price, USD/barrel",
    "T10YIE": "10-year breakeven inflation rate, % (from 2003)",
    "DTWEXBGS": "Nominal broad US dollar index (from 2006)",
    "USREC": "NBER recession indicator (monthly, 0/1)",
}

for _d in (RAW_DIR, PROC_DIR, RESULTS_DIR, FIG_DIR, GEN_DIR):
    _d.mkdir(parents=True, exist_ok=True)
