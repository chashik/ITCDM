"""Lab 0. Research domain: data inventory, quality checks, stylised facts.

Run:  python -m lab0_domain.run
Produces figures  paper/figures/lab0_*.pdf  and tables/macros in paper/generated.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.stattools import acf

from dmfin import config as C
from dmfin import plotting as P
from dmfin.data import load_credit, load_fred_panel, load_market_panel, load_stock_prices, load_universe
from dmfin.features import add_targets, market_features
from dmfin.reporting import save_numbers, save_table
from dmfin.splits import chrono_split


def inventory(panel, fred, stocks, credit):
    rows = []
    rows.append(("Yahoo Finance", "^GSPC (S&P 500 OHLCV)", "daily",
                 panel.index.min(), panel.index.max(), len(panel), 0.0))
    from dmfin.data import download_fred
    for s, desc in C.FRED_SERIES.items():
        x = download_fred(s).loc[C.START:C.END]   # own calendar of the series
        freq = "monthly" if s == "USREC" else "daily"
        first = x.first_valid_index()
        rows.append(("FRED", s, freq, first, x.last_valid_index(),
                     int(x.notna().sum()), 100 * x.loc[first:].isna().mean()))
    rows.append(("Yahoo Finance", f"{stocks.shape[1]} stocks, adj. close", "daily",
                 stocks.index.min(), stocks.index.max(), stocks.size,
                 100 * stocks.isna().mean().mean()))
    rows.append(("UCI ML Repository", "Default of credit card clients", "cross-section",
                 pd.NaT, pd.NaT, len(credit), 100 * credit.isna().mean().mean()))
    df = pd.DataFrame(rows, columns=["Source", "Series", "Frequency", "First", "Last",
                                     "Observations", "Missing, %"])
    df["First"] = [d.strftime("%Y-%m-%d") if pd.notna(d) else "2005" for d in df["First"]]
    df["Last"] = [d.strftime("%Y-%m-%d") if pd.notna(d) else "2005" for d in df["Last"]]
    return df


def main():
    panel = load_market_panel()
    fred = load_fred_panel()
    stocks = load_stock_prices()
    universe = load_universe()
    credit = load_credit()
    feats = add_targets(market_features(panel))

    inv = inventory(panel, fred, stocks, credit)
    save_table("lab0_inventory", inv.set_index("Source"), nd={"Missing, %": 2},
               col_format="p{2.6cm}p{4.1cm}ccccc")

    # ---------------------------------------------------------------- quality
    same_open = (panel["open"] == panel["close"].shift()).groupby(panel.index.year).mean()
    f, ax = P.fig(2.4)
    ax.bar(same_open.index, 100 * same_open.values, color=P.BLUE, width=0.75)
    ax.set_ylabel("share of days, %")
    ax.set_xlabel("year")
    P.save(f, "lab0_open_quality")

    # ------------------------------------------------------------ index level
    f, ax = P.fig(3.0)
    P.shade_recessions(ax, panel["USREC"])
    P.shade_splits(ax)
    ax.plot(panel.index, panel["adj_close"], color=P.BLUE, lw=1.0)
    ax.set_yscale("log")
    ax.set_ylabel("index points (log scale)")
    from matplotlib.ticker import FixedLocator, NullFormatter, ScalarFormatter
    ax.yaxis.set_major_locator(FixedLocator([700, 1000, 1500, 2000, 3000, 5000, 7000]))
    ax.yaxis.set_major_formatter(ScalarFormatter()); ax.yaxis.set_minor_formatter(NullFormatter())
    for d, txt, dx, dy in [("2009-03-09", "GFC low", 40, -4), ("2020-03-23", "COVID-19", 0, -18),
                           ("2022-10-12", "2022 bear", 0, -18)]:
        y = panel["adj_close"].loc[d]
        ax.annotate(txt, (pd.Timestamp(d), y), xytext=(dx, dy), textcoords="offset points",
                    ha="center", fontsize=9, color=P.INK2,
                    arrowprops=dict(arrowstyle="-", color=P.MUTED, lw=0.6))
    P.save(f, "lab0_spx")

    # ---------------------------------------------------- macro small multiples
    f, axs = P.fig(5.6, nrows=4, sharex=True)
    series = [(feats["vix"], "VIX, points"),
              (feats["rv_22"] * 100, "realized vol. 22d, %"),
              (feats["credit_baa_aaa"], "Baa - Aaa spread, p.p."),
              (feats["term_10y3m"], "10y - 3m term spread, p.p.")]
    for ax, (s, lab) in zip(axs, series):
        P.shade_recessions(ax, panel["USREC"])
        ax.plot(s.index, s.values, color=P.BLUE, lw=0.8)
        ax.set_ylabel(lab, fontsize=9)
    axs[-1].axhline(0, color=P.INK2, lw=0.6)
    P.save(f, "lab0_macro")

    # ---------------------------------------------------------- stylised facts
    r = feats["ret"].dropna()
    z = (r - r.mean()) / r.std()
    f, axs = P.fig(2.8, ncols=2)
    grid = np.linspace(-8, 8, 400)
    axs[0].hist(z, bins=160, range=(-8, 8), density=True, color=P.BLUE, alpha=0.85, label="S&P 500")
    axs[0].plot(grid, stats.norm.pdf(grid), color=P.ORANGE, lw=1.4, label="N(0,1)")
    axs[0].set_yscale("log"); axs[0].set_ylim(1e-4, 1)
    axs[0].set_xlabel("standardized daily log-return"); axs[0].set_ylabel("density (log scale)")
    axs[0].legend(loc="upper right")
    lags = np.arange(1, 61)
    a1 = acf(r, nlags=60)[1:]
    a2 = acf(r.abs(), nlags=60)[1:]
    ci = 1.96 / np.sqrt(len(r))
    axs[1].axhspan(-ci, ci, color="#d9d8d3", alpha=0.6, lw=0)
    axs[1].plot(lags, a1, "o", ms=2.5, color=P.BLUE, label="returns $r_t$")
    axs[1].plot(lags, a2, "o", ms=2.5, color=P.ORANGE, label="absolute returns $|r_t|$")
    axs[1].set_xlabel("lag, trading days"); axs[1].set_ylabel("autocorrelation")
    axs[1].legend(loc="upper right")
    P.save(f, "lab0_stylised")

    # ------------------------------------------------------------- splits plot
    sp = chrono_split(feats.dropna(subset=["y_rv_22", "pk_m"]).index, horizon=22)
    # ------------------------------------------------- correlation of features
    cols = ["ret_22", "pk_d", "pk_w", "pk_m", "rv_22", "vix", "vix_chg_5", "vrp",
            "term_10y3m", "credit_baa_aaa", "dgs10_chg_22", "oil_ret_22", "volume_z", "drawdown"]
    corr = feats.loc[: C.TRAIN_END, cols].corr()
    f, ax = P.fig(5.0, w=5.6)
    im = ax.imshow(corr.values, cmap=P.DIV_CELL, vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, rotation=60, ha="right", fontsize=9)
    ax.set_yticks(range(len(cols))); ax.set_yticklabels(cols, fontsize=9)
    ax.grid(False)
    for i in range(len(cols)):
        for j in range(len(cols)):
            ax.text(j, i, P.per_cent(corr.values[i, j]), ha="center", va="center", fontsize=9, color=P.INK)
    cb = f.colorbar(im, ax=ax, fraction=0.04, pad=0.02, label="Pearson correlation (training block), multiplied by 100")
    cb.set_ticks([-1, -0.5, 0, 0.5, 1], labels=[P.per_cent(v) for v in (-1, -0.5, 0, 0.5, 1)])
    P.save(f, "lab0_corr")

    # --------------------------------------------------------- credit summary
    cdesc = credit.describe().T[["mean", "std", "min", "50%", "max"]]
    keep = ["LIMIT_BAL", "AGE", "PAY_1", "BILL_AMT1", "PAY_AMT1", "default"]
    cdesc.index.name = "Attribute"
    save_table("lab0_credit_summary", cdesc.loc[keep], nd={"mean": 2, "std": 2, "min": 0, "50%": 0, "max": 0})

    n = {
        "n_days": len(panel), "first_day": panel.index.min().strftime("%Y-%m-%d"),
        "last_day": panel.index.max().strftime("%Y-%m-%d"),
        "n_train": len(sp.train), "n_val": len(sp.val), "n_test": len(sp.test),
        "ret_mean_ann": float(r.mean() * 252 * 100), "ret_vol_ann": float(r.std() * np.sqrt(252) * 100),
        "ret_skew": float(stats.skew(r)), "ret_kurt": float(stats.kurtosis(r)),
        "ret_min": float(r.min() * 100), "ret_min_date": r.idxmin().strftime("%Y-%m-%d"),
        "ret_max": float(r.max() * 100), "ret_max_date": r.idxmax().strftime("%Y-%m-%d"),
        "share_beyond_4sd": float(100 * (z.abs() > 4).mean()),
        "normal_beyond_4sd": float(100 * 2 * stats.norm.sf(4)),
        "acf_abs_lag_1": float(a2[0]), "acf_abs_lag_60": float(a2[-1]),
        "acf_ret_lag_1": float(a1[0]),
        "same_open_2000_2005": float(100 * same_open.loc[2000:2005].mean()),
        "same_open_2016_2026": float(100 * same_open.loc[2016:2026].mean()),
        "credit_n": len(credit), "credit_default_rate": float(100 * credit["default"].mean()),
        "credit_n_features": credit.shape[1] - 1,
        "n_stocks": stocks.shape[1], "n_stock_days": stocks.shape[0],
        "n_sectors": universe.nunique(),
        "vix_max": float(panel["VIXCLS"].max()), "vix_max_date": panel["VIXCLS"].idxmax().strftime("%Y-%m-%d"),
        "corr_vix_rv": float(corr.loc["vix", "rv_22"]),
        "corr_pkm_rv": float(corr.loc["pk_m", "rv_22"]),
        "oil_min": float(panel["DCOILWTICO"].min()),
    }
    save_numbers("lab0", n)
    print(n)


if __name__ == "__main__":
    main()
