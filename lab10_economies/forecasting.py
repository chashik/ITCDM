"""Question 4: do advanced data mining methods improve on simple ones for these data?

Task: forecast inflation (CPI, y/y) 1 and 3 months ahead, from every month-end 2023-01 .. 2025-12 minus the horizon,
with models fitted only on the months whose outcome was known at that time (expanding window). Every model predicts the
change of inflation over the horizon (tree models cannot extrapolate a level); "random walk" predicts no change.

Two panels
  short  2021-2025, all features of the collection: own inflation and its lag, the month-on-month CPI change, food and
         petrol price changes (3-month means), the US dollar rate (y/y), policy and mortgage rates, the real policy rate
  long   2013-2025, fewer features but eleven more years: inflation and two lags, policy rate, its 3-month change, the
         US dollar rate (y/y), the real policy rate. Sources: Belstat CPI index (2010 = 100), the official GUS CPI index for Poland (via OECD),
         Rosstat CPI (EEC, y/y), Lithuanian HICP (Eurostat); IMF MFS_IR policy rates for Belarus and Russia (Russia from
         2020-12: Bank of Russia), NBP reference rate archive for Poland, ECB deposit facility rate for Lithuania (euro
         since 2015; before, a currency board pegged to the euro); IMF ER exchange rates, ECB for Lithuania
A feature with fewer than 24 months in a country (Russian petrol: December values only) is left out for that country.
Models: random walk; AR (own inflation and lag, OLS); ridge; lasso; random forest; gradient boosting; k nearest
neighbours; "pooled" ridge and random forest trained on all four countries at once (features available in every
country, one indicator per country).

Outputs: q4_forecasts.csv (RMSE relative to the random walk, Diebold-Mariano p against it), q4_paths.csv (the
forecasts one month ahead of every model and origin), q4_mutual_information.csv,
q4_forecast_skill.png.
Run (from code/):  python -m lab10_economies.forecasting
"""
import json
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import LassoCV, LinearRegression, RidgeCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .common import COUNTRIES, P, panel, save_figure, save_table
from .paths import RAW
from .prepare import fx
ORIGINS = ("2023-01", "2025-12")
SHORT = ["y", "y_lag1", "cpi_mom", "food_3m", "petrol_3m", "fx_yoy", "policy", "mortgage", "real_policy"]
LONG = ["y", "y_lag1", "y_lag2", "policy", "policy_3m", "fx_yoy", "real_policy"]


def short_panel(c: str) -> pd.DataFrame:
    p = panel(c)
    food = p[["price_flour_wheat", "price_milk", "price_sugar"]].pct_change().mean(axis=1) * 100
    return pd.DataFrame({"y": p["cpi_yoy"], "y_lag1": p["cpi_yoy"].shift(1), "cpi_mom": p["cpi_mom"],
                         "food_3m": food.rolling(3).mean(), "petrol_3m": (100 * p["price_petrol"].pct_change()).rolling(3).mean(),
                         "fx_yoy": p["fx_usd_yoy"], "policy": p["policy_rate"], "mortgage": p["mortgage_rate"],
                         "real_policy": p["real_policy_rate"]})


def _imf_series(path: Path, indicator: str) -> pd.Series:
    s = path.read_text(encoding="utf-8")
    for attrs, body in re.findall(r"<Series ([^>]*)>(.*?)</Series>", s, re.S):
        if dict(re.findall(r'(\w+)="([^"]*)"', attrs)).get("INDICATOR", "").startswith(indicator):
            obs = [dict(re.findall(r'(\w+)="([^"]*)"', o)) for o in re.findall(r"<Obs ([^>]*)/>", body)]
            return pd.Series({pd.Period(o["TIME_PERIOD"].replace("-M", "-"), "M"): float(o["OBS_VALUE"]) for o in obs})
    raise ValueError(indicator)


def long_panel(c: str) -> pd.DataFrame:
    if c == "BY":
        x = pd.read_excel(RAW / "belstat" / "IPC-0926.xlsx", sheet_name="2010 г. = 100", header=None, dtype=str)
        lev = {}
        for _, r in x.iterrows():
            if re.fullmatch(r"\d{4}", str(r[0]).strip()):
                for mo in range(1, 13):
                    v = str(r[mo]).replace(",", ".")
                    if re.fullmatch(r"[\d.]+", v):
                        lev[pd.Period(f"{int(str(r[0]).strip())}-{mo:02d}", "M")] = float(v)
        lev = pd.Series(lev).sort_index()
        y = 100 * (lev / lev.shift(12) - 1)
        pol = _imf_series(RAW / "imf" / "mfs_ir_BLR_long.xml", "MFS166")
        e = fx("BLR")
    elif c in ("RU", "LT"):
        y, pol, e = _long_ru_lt(c)
    else:
        o = pd.read_csv(RAW / "oecd" / "cpi_pol_IX__Z.csv")     # official CPI of GUS, index 2015 = 100
        lev = pd.Series({pd.Period(t, "M"): v for t, v in zip(o["TIME_PERIOD"], o["OBS_VALUE"])}).sort_index()
        y = 100 * (lev / lev.shift(12) - 1)
        xml = (RAW / "nbp" / "stopy_archiwum.xml").read_text(encoding="utf-8-sig")
        ev = {pd.Timestamp(d): float(v.replace(",", ".")) for d, body in re.findall(r'obowiazuje_od="([\d-]+)">(.*?)</pozycje>', xml, re.S)
              for v in re.findall(r'id="ref"\s+oprocentowanie="([\d,]+)"', body)}
        days = pd.Series(ev).sort_index().reindex(pd.date_range("1998-01-01", "2026-12-31")).ffill()
        pol = days.groupby(days.index.to_period("M")).last()
        e = fx("POL")
    e.index = pd.PeriodIndex(e.index, freq="M")
    d = pd.DataFrame({"y": y, "policy": pol, "fx": e}).loc["2012-01":"2025-12"]
    return pd.DataFrame({"y": d["y"], "y_lag1": d["y"].shift(1), "y_lag2": d["y"].shift(2), "policy": d["policy"],
                         "policy_3m": d["policy"].diff(3), "fx_yoy": 100 * (d["fx"] / d["fx"].shift(12) - 1),
                         "real_policy": d["policy"] - d["y"]}).loc["2013-01":]


def _eec_yoy(country: str = "Россия") -> pd.Series:
    """EEC Series_CPI/CPI_3.xls: CPI to the same month of the previous year, %, by country (Rosstat for Russia)."""
    d = pd.read_excel(RAW / "eec" / "CPI_3.xls", header=None)
    out, year = {}, None
    for _, r in d.iterrows():
        m = re.match(r"(\d{4}) год", str(r[0]).strip())
        if m:
            year = int(m.group(1))
        elif year and str(r[0]).strip() == country:
            out.update({pd.Period(f"{year}-{mo:02d}", "M"): float(r[mo]) - 100 for mo in range(1, 13) if pd.notna(r[mo])})
    return pd.Series(out).sort_index()


def _ecb_month_end(path: Path) -> pd.Series:
    e = pd.read_csv(path)
    days = pd.Series(e["OBS_VALUE"].to_numpy(), index=pd.to_datetime(e["TIME_PERIOD"])).sort_index()
    return days.groupby(days.index.to_period("M")).last()


def _long_ru_lt(c: str) -> tuple[pd.Series, pd.Series, pd.Series]:
    recent = panel(c, base=True)                               # collection from 2020-12 (Bank of Russia, ECB)
    if c == "RU":
        y = _eec_yoy()
        pol = recent["policy_rate"].combine_first(_imf_series(RAW / "imf" / "mfs_ir_RUS_long.xml", "MFS166"))
        e = fx("RUS")
    else:
        d = json.loads((RAW / "eurostat" / "prc_hicp_manr_LT_2011.json").read_text())
        inv = {v: k for k, v in d["dimension"]["time"]["category"]["index"].items()}
        y = pd.Series({pd.Period(inv[int(i)], "M"): v for i, v in d["value"].items()}).sort_index()
        pol = _ecb_month_end(RAW / "ecb" / "dfr_long.csv")
        x = pd.read_csv(RAW / "ecb" / "exr_usd_long.csv")
        e = 1 / pd.Series(x["OBS_VALUE"].to_numpy(), index=x["TIME_PERIOD"]).sort_index()   # euro per US dollar
    return y, pol, e


def models() -> dict:
    return {
        "AR (own lags)": None,                              # handled separately: OLS on y and y_lag1
        "ridge": lambda: make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 30))),
        "lasso": lambda: make_pipeline(StandardScaler(), LassoCV(cv=TimeSeriesSplit(3), alphas=30, max_iter=50000)),
        "random forest": lambda: RandomForestRegressor(n_estimators=300, min_samples_leaf=3, random_state=0, n_jobs=1),
        "gradient boosting": lambda: GradientBoostingRegressor(n_estimators=150, max_depth=2, learning_rate=0.05,
                                                               subsample=0.8, random_state=0),
        "k nearest neighbours": lambda: make_pipeline(StandardScaler(), KNeighborsRegressor(5)),
    }


def usable(d: pd.DataFrame, feats: list) -> list:
    return [f for f in feats if d[f].notna().sum() >= 24]


def evaluate(panels: dict, feats: list, h: int) -> pd.DataFrame:
    """Expanding-window forecasts of every model; returns errors per country, model and origin."""
    rows = []
    origins = pd.period_range(*ORIGINS, freq="M")
    own = {c: usable(d, feats) for c, d in panels.items()}
    common = [f for f in feats if all(f in own[c] for c in panels)]
    codes = list(panels)
    dummies = lambda c: {f"is_{k}": float(c == k) for k in codes[1:]}     # one indicator per country (first = base)
    data = {}
    for c, d in panels.items():
        delta = d["y"].shift(-h) - d["y"]
        data[c] = (d[own[c]], d[common].assign(**dummies(c)), delta, d["y"])
    for t in origins:
        if any(t + h > d.index[-1] for d in panels.values()):
            continue
        pooled_X, pooled_y = [], []
        for c, (X, XC, delta, y) in data.items():
            ok = (XC.index + h <= t) & XC.notna().all(axis=1).to_numpy() & delta.notna().to_numpy()
            pooled_X.append(XC[ok]); pooled_y.append(delta[ok])
        PX, Py = pd.concat(pooled_X), pd.concat(pooled_y)
        for c, (X, XC, delta, y) in data.items():
            ok = (X.index + h <= t) & X.notna().all(axis=1).to_numpy() & delta.notna().to_numpy()
            Xt, yt, xnow = X[ok], delta[ok], X.loc[[t]]
            actual = y[t + h]
            preds = {"random walk": 0.0}
            preds["AR (own lags)"] = float(LinearRegression().fit(Xt[["y", "y_lag1"]], yt).predict(xnow[["y", "y_lag1"]])[0])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                for name, make in models().items():
                    if make is not None:
                        preds[name] = float(make().fit(Xt, yt).predict(xnow)[0])
                preds["pooled ridge"] = float(models()["ridge"]().fit(PX, Py).predict(XC.loc[[t]])[0])
                preds["pooled random forest"] = float(models()["random forest"]().fit(PX, Py).predict(XC.loc[[t]])[0])
            for name, dhat in preds.items():
                rows.append({"country": c, "origin": str(t), "model": name, "error": y[t] + dhat - actual,
                             "actual": actual})
    return pd.DataFrame(rows)


def dm_pvalue(e_model: np.ndarray, e_rw: np.ndarray, h: int) -> float:
    """Diebold-Mariano test of equal squared error against the random walk (two-sided, HAC variance with h-1 lags)."""
    d = e_rw ** 2 - e_model ** 2
    n = len(d)
    gamma = [np.mean((d[k:] - d.mean()) * (d[:n - k] - d.mean())) for k in range(h)]
    var = (gamma[0] + 2 * sum(gamma[1:])) / n
    if var <= 0:
        return np.nan
    return float(2 * stats.t.sf(abs(d.mean() / np.sqrt(var)), n - 1))


def summary(err: pd.DataFrame, panel_name: str, h: int) -> pd.DataFrame:
    out = []
    for c, g in err.groupby("country"):
        rw = g[g.model == "random walk"].set_index("origin")["error"]
        for m, gm in g.groupby("model", sort=False):
            e = gm.set_index("origin")["error"].reindex(rw.index)
            out.append({"panel": panel_name, "horizon": h, "country": c, "model": m, "forecasts": len(e),
                        "RMSE, p.p.": float(np.sqrt(np.mean(e ** 2))),
                        "RMSE relative to random walk": float(np.sqrt(np.mean(e ** 2)) / np.sqrt(np.mean(rw ** 2))),
                        "DM p vs random walk": np.nan if m == "random walk" else dm_pvalue(e.to_numpy(), rw.to_numpy(), h)})
    return pd.DataFrame(out)


def mutual_information(panels: dict, h: int = 3) -> pd.DataFrame:
    """Dependence of the change of inflation over h months on each feature: k-NN mutual information against the mutual
    information that a linear Gaussian relation with the same correlation would have (-0.5 ln(1 - r^2)); the p-value
    compares the estimate with its values for the target shifted cyclically by 6 or more months (keeps autocorrelation)."""
    rows = []
    for c, d in panels.items():
        delta = (d["y"].shift(-h) - d["y"]).rename("target")
        fs = usable(d, SHORT)
        z = pd.concat([d[fs], delta], axis=1).dropna()
        mi = mutual_info_regression(z[fs], z["target"], n_neighbors=5, random_state=0)
        tgt = z["target"].to_numpy()                       # the k-NN estimate is biased upwards on few points; its value
        null = np.array([mutual_info_regression(z[fs], np.roll(tgt, k), n_neighbors=5, random_state=0)
                         for k in range(6, len(tgt) - 5)])  # without dependence: target shifted cyclically, which keeps
        #                                                    its autocorrelation (a plain permutation would not)
        for k, (f, m) in enumerate(zip(fs, mi)):
            r = z[f].corr(z["target"])
            rows.append({"country": c, "feature": f, "months": len(z), "correlation": r,
                         "MI, nats (k-NN)": m, "MI of a linear relation, nats": -0.5 * np.log(1 - r ** 2),
                         "cyclic-shift p": float((null[:, k] >= m).mean())})
    return pd.DataFrame(rows)


def main():
    short = {c: short_panel(c) for c in COUNTRIES}
    long = {c: long_panel(c) for c in COUNTRIES}
    for c in COUNTRIES:
        print(c, "long panel", long[c].index[0], "..", long[c].index[-1], "complete rows", int(long[c].notna().all(axis=1).sum()))
    res, paths = [], []
    for name, pn, feats in (("short 2021-2025", short, SHORT), ("long 2013-2025", long, LONG)):
        for h in (1, 3):
            err = evaluate(pn, feats, h)
            res.append(summary(err, name, h))
            if h == 1:                                  # the forecasts themselves, for the figure of forecast paths
                paths.append(err.assign(panel=name, forecast=err["actual"] + err["error"]))
            print("done", name, h, flush=True)
    res = pd.concat(res)
    save_table(pd.concat(paths).set_index("panel"), "q4_paths", 4)
    save_table(res.set_index("panel"), "q4_forecasts", 3)
    mi = mutual_information(short)
    save_table(mi.set_index("country"), "q4_mutual_information", 3)

    f, axs = P.fig(6.4, nrows=2, ncols=2, sharey=True)
    settings = [("short 2021-2025", 1), ("short 2021-2025", 3), ("long 2013-2025", 1), ("long 2013-2025", 3)]
    model_order = [m for m in res["model"].unique() if m != "random walk"]
    for ax, c in zip(axs.ravel(), COUNTRIES):
        for i, (pn, h) in enumerate(settings):
            r = res[(res.country == c) & (res.panel == pn) & (res.horizon == h)].set_index("model")
            ax.barh(np.arange(len(model_order)) + (i - 1.5) * 0.2, r.loc[model_order, "RMSE relative to random walk"],
                    height=0.2, color=P.CAT[i], label=f"{pn.split()[0]} panel, {h} month{'s' if h > 1 else ''} ahead")
        ax.axvline(1, color=P.INK, lw=0.8)
        ax.set_yticks(range(len(model_order)), model_order, fontsize=9)
        ax.set_title(COUNTRIES[c], fontsize=9.5)
        ax.set_xlabel("RMSE relative to no change (< 1: better)")
    axs[0, 0].invert_yaxis()
    h, l = axs[0, 1].get_legend_handles_labels()
    f.legend(h, l, loc="lower center", ncol=2, fontsize=9, frameon=False)
    f.tight_layout(rect=(0, 0.07, 1, 1))
    save_figure(f, "q4_forecast_skill")
    pd.set_option("display.width", 220)
    print(res.round(3).to_string(index=False)); print(mi.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
