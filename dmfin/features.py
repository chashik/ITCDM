"""Feature engineering for the S&P 500 market panel.

Conventions
-----------
* A feature dated t uses information available at the close of day t only; series that are
  published after the date they refer to enter with their publication delay
  (``dmfin.data.PUBLICATION_LAG_DAYS``).
* A target dated t describes the future, days t+1 .. t+h.
* Log-returns: r_t = ln(P_t / P_{t-1}).
* Daily variance proxy: Parkinson (1980) range estimator
  sigma^2_t = (ln H_t - ln L_t)^2 / (4 ln 2).  The open price is not used:
  in the public data the S&P 500 open equals the previous close for most
  days before 2006 (see Lab 0), which would bias open-based estimators.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import TRADING_DAYS
from .data import PUBLICATION_LAG_DAYS, available_at

LN2x4 = 4.0 * np.log(2.0)


def log_returns(price: pd.Series) -> pd.Series:
    return np.log(price).diff()


def parkinson_var(high: pd.Series, low: pd.Series) -> pd.Series:
    return np.log(high / low) ** 2 / LN2x4


def future_realized_vol(r: pd.Series, h: int) -> pd.Series:
    """Annualised realised volatility over days t+1..t+h (target at date t)."""
    rv = (r ** 2).rolling(h).sum().shift(-h)
    return np.sqrt(TRADING_DAYS / h * rv)


def past_realized_vol(r: pd.Series, h: int) -> pd.Series:
    """Annualised realised volatility over days t-h+1..t (feature at date t)."""
    return np.sqrt(TRADING_DAYS / h * (r ** 2).rolling(h).sum())


def market_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Daily feature table used by Labs 2, 5, 6, 7 and 8."""
    p = panel
    f = pd.DataFrame(index=p.index)
    r = log_returns(p["adj_close"])
    pk = parkinson_var(p["high"], p["low"])
    f["ret"] = r
    f["ret_5"] = r.rolling(5).sum()
    f["ret_22"] = r.rolling(22).sum()
    f["ret_66"] = r.rolling(66).sum()
    # HAR components (Corsi, 2009) built on the Parkinson variance, annualised
    f["pk_d"] = np.sqrt(TRADING_DAYS * pk)
    f["pk_w"] = np.sqrt(TRADING_DAYS * pk.rolling(5).mean())
    f["pk_m"] = np.sqrt(TRADING_DAYS * pk.rolling(22).mean())
    f["rv_22"] = past_realized_vol(r, 22)
    f["vix"] = p["VIXCLS"]
    f["vix_chg_5"] = np.log(p["VIXCLS"]).diff(5)
    f["vrp"] = p["VIXCLS"] / 100 - f["rv_22"]          # variance-risk-premium proxy
    f["term_10y3m"] = p["DGS10"] - p["DTB3"]
    f["term_10y2y"] = p["DGS10"] - p["DGS2"]
    f["credit_baa_aaa"] = p["DBAA"] - p["DAAA"]
    f["credit_baa_10y"] = p["DBAA"] - p["DGS10"]
    f["dgs10_chg_22"] = p["DGS10"].diff(22)
    f["ff_rate"] = available_at(p["DFF"], p.index, PUBLICATION_LAG_DAYS["DFF"])
    f["oil_ret_22"] = p["DCOILWTICO"].pct_change(22, fill_method=None).clip(-1, 1)
    f["breakeven_10y"] = p["T10YIE"]            # starts 2003
    usd = available_at(p["DTWEXBGS"], p.index, PUBLICATION_LAG_DAYS["DTWEXBGS"])
    f["usd_ret_22"] = np.log(usd).diff(22)            # starts 2006
    lv = np.log(p["volume"].where(p["volume"] > 0))
    f["volume_z"] = (lv - lv.rolling(252, min_periods=200).mean()) / \
        lv.rolling(252, min_periods=200).std()
    f["drawdown"] = p["adj_close"] / p["adj_close"].cummax() - 1
    f["recession"] = p["USREC"]
    return f


def add_targets(f: pd.DataFrame, horizons=(1, 5, 22)) -> pd.DataFrame:
    out = f.copy()
    for h in horizons:
        out[f"y_rv_{h}"] = future_realized_vol(f["ret"], h)
        out[f"y_logrv_{h}"] = np.log(out[f"y_rv_{h}"])
        out[f"y_ret_{h}"] = f["ret"].rolling(h).sum().shift(-h)
    return out
