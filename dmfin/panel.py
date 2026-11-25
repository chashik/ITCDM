"""Many-series panel for global models (iteration 2): one row per (series, day).

Series: the S&P 500, twelve further markets (D1) and the 49 Fama-French
industry portfolios (D3), each on its own trading calendar.  For every series
and day t the panel holds

* own information (known at the close of t): the daily log-return and the
  heterogeneous-autoregressive components of realized volatility computed from
  squared returns over 1, 5 and 22 days (log scale);
* common market information: log VIX, its daily change, the Baa-Aaa credit
  spread, the 10y-3m term spread and the St. Louis Fed stress index (used only
  after its publication, seven days after its reference date);
* the target: log realized volatility of the next ``h`` days of the series.

Time zones: US market data of day t are known only at the New York close.
Markets that close earlier (Tokyo, London, Euro area) therefore receive the
US data of the previous US trading day; otherwise a model would use
information from the future of these markets.
"""
from __future__ import annotations

import functools

import numpy as np
import pandas as pd

from . import config as C
from .data import PUBLICATION_LAG_DAYS, available_at, download_fred, load_market_panel
from .data_ext import MULTI_ASSET, load_ff_industries, load_multi_asset

NON_US = {"^FTSE", "^N225", "^STOXX50E"}
OWN = ["ret", "lrv_d", "lrv_w", "lrv_m"]
COMMON = ["log_vix", "d_log_vix", "credit", "term", "stress"]
FEATURES = OWN + COMMON
TD = 252.0
EPS = 1e-4


def _own_features(r: pd.Series, h: int) -> pd.DataFrame:
    r2 = r ** 2
    f = pd.DataFrame(index=r.index)
    f["ret"] = r
    f["lrv_d"] = np.log(np.sqrt(TD * r2) + EPS)
    f["lrv_w"] = np.log(np.sqrt(TD * r2.rolling(5).mean()) + EPS)
    f["lrv_m"] = np.log(np.sqrt(TD * r2.rolling(22).mean()) + EPS)
    fut = r2.rolling(h).sum().shift(-h)
    f[f"y_{h}"] = np.log(np.sqrt(TD / h * fut) + EPS)
    return f


def _common() -> pd.DataFrame:
    p = load_market_panel()
    c = pd.DataFrame(index=p.index)
    c["log_vix"] = np.log(p["VIXCLS"])
    c["d_log_vix"] = c["log_vix"].diff()
    c["credit"] = p["DBAA"] - p["DAAA"]
    c["term"] = p["DGS10"] - p["DTB3"]
    st = download_fred("STLFSI4")                       # weekly, dated on Fridays, published the next Thursday
    c["stress"] = available_at(st, c.index, PUBLICATION_LAG_DAYS["STLFSI4"])
    return c


@functools.lru_cache(maxsize=4)
def build_panel(h: int = 5) -> pd.DataFrame:
    """Long panel with columns series, date, group, FEATURES, y_h (rows with a complete history only)."""
    cache = C.PROC_DIR / f"panel_h{h}.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    returns = {"^GSPC": np.log(load_market_panel()["adj_close"]).diff()}
    for t, d in load_multi_asset().items():
        if t != "^GSPC":
            returns[t] = np.log(d["adj_close"]).diff()
    for name, r in load_ff_industries().items():
        returns[f"FF:{name}"] = np.log1p(r)
    common = _common()
    parts = []
    for s, r in returns.items():
        r = r.dropna()
        r = r[(r.index >= pd.Timestamp(C.START)) & (r.index <= pd.Timestamp(C.END))]
        f = _own_features(r, h)
        # common US information known at the close of each day of this series
        cm = common.reindex(common.index.union(f.index)).sort_index().ffill(limit=5)
        if s in NON_US:
            cm = cm.shift(1)                             # previous US close only
        f = f.join(cm.reindex(f.index))
        f["series"] = s
        f["group"] = "industry" if s.startswith("FF:") else ("index" if s.startswith("^") else "asset")
        parts.append(f.dropna(subset=FEATURES))
    panel = pd.concat(parts).reset_index().rename(columns={"index": "date"})
    panel = panel.rename(columns={panel.columns[0]: "date"}) if "date" not in panel.columns else panel
    panel = panel.sort_values(["series", "date"]).reset_index(drop=True)
    panel.to_parquet(cache)
    return panel


def split_masks(panel: pd.DataFrame, h: int = 5):
    """Boolean masks of the chronological blocks; the last h rows of a block are purged per series."""
    d = panel["date"]
    tr = d <= pd.Timestamp(C.TRAIN_END)
    va = (d > pd.Timestamp(C.TRAIN_END)) & (d <= pd.Timestamp(C.VAL_END))
    te = d > pd.Timestamp(C.VAL_END)
    y = panel[f"y_{h}"].notna()
    out = {}
    for name, m in (("train", tr), ("val", va), ("test", te)):
        mm = m & y
        # purge: the target of the last h days of the block reaches into the next block
        last = panel[mm].groupby("series")["date"].transform(lambda s: s.rank(ascending=False) <= h)
        purge = pd.Series(False, index=panel.index)
        purge.loc[last[last].index] = True
        out[name] = (mm & ~purge).to_numpy()
    return out


def window_index(panel: pd.DataFrame, mask: np.ndarray, lookback: int) -> np.ndarray:
    """Row numbers t of the panel that have ``lookback`` rows of the same series ending at t."""
    pos_in_series = panel.groupby("series").cumcount().to_numpy()
    return np.where(mask & (pos_in_series >= lookback - 1))[0]


def gather_windows(X: np.ndarray, rows: np.ndarray, lookback: int) -> np.ndarray:
    """Windows X[t-L+1..t] for the rows (works for numpy arrays and torch tensors)."""
    offs = np.arange(-lookback + 1, 1)
    return X[rows[:, None] + offs[None, :]]


def names() -> dict[str, str]:
    out = dict(MULTI_ASSET)
    out.update({s: s[3:] for s in build_panel()["series"].unique() if s.startswith("FF:")})
    return out
