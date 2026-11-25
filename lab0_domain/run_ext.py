"""Lab 0 (iteration 2). Extended domain data: inventory, coverage, quality checks.

Run:  python -m lab0_domain.run_ext
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from dmfin import config as C
from dmfin import plotting as P
from dmfin.data import load_market_panel, load_universe
from dmfin.data_ext import (FREDMD_VINTAGE, MULTI_ASSET, VOL_FRED, VOL_YAHOO, download_fred, download_yahoo,
                            inventory, load_edgar_meta, load_ff49_sic, load_ff_factors, load_ff_industries,
                            load_fin_tweets, load_fomc_dates, load_fredmd, load_german_credit, load_multi_asset,
                            sic_to_ff49)
from dmfin.panel import build_panel, split_masks
from dmfin.reporting import save_numbers, save_table

SHORT = {"^GSPC": "S&P 500", "^IXIC": "Nasdaq", "^RUT": "Russell 2000", "^FTSE": "FTSE 100", "^N225": "Nikkei 225",
         "^STOXX50E": "Euro Stoxx 50", "GC=F": "Gold", "CL=F": "WTI oil", "TLT": "Treasury 20y+",
         "IEF": "Treasury 7-10y", "HYG": "High yield", "EEM": "Emerging mkts", "DX-Y.NYB": "Dollar index"}


def main():
    num = {"fredmd_vintage": FREDMD_VINTAGE}
    inv = inventory()
    tab = inv.copy()
    tab["first"] = tab["first"].dt.strftime("%Y-%m-%d").fillna("--")
    tab["last"] = tab["last"].dt.strftime("%Y-%m-%d").fillna("--")
    tab["missing"] = 100 * tab["missing"]
    tab = tab.rename(columns={"id": "Id", "source": "Data", "frequency": "Frequency", "series": "Series",
                              "first": "First", "last": "Last", "missing": "Missing, \\%"}).set_index("Id")
    save_table("lab0_ext_inventory", tab, nd={"Series": 0, "Missing, \\%": 1}, col_format="lp{4.6cm}lcccc")

    # ------------------------------------------------ quality checks
    ma = load_multi_asset()
    wti = download_yahoo("CL=F").loc[C.START:C.END]
    neg = wti[(wti[["close", "low"]] <= 0).any(axis=1)]
    num.update({"wti_neg_n": len(neg), "wti_neg_date": neg.index.min().strftime("%Y-%m-%d") if len(neg) else "--",
                "wti_neg_low": float(wti["low"].min())})
    spx_days = ma["^GSPC"].index
    for t, key in (("^N225", "nikkei"), ("^FTSE", "ftse")):
        other = ma[t].index
        num[f"{key}_closed_share"] = 100 * (~spx_days.isin(other)).mean()
    ind = load_ff_industries()
    num.update({"n_industries": ind.shape[1], "ind_min_ret": float(ind.min().min()),
                "ind_min_name": ind.min().idxmin(), "ind_min_date": ind.min(axis=1).idxmin().strftime("%Y-%m-%d"),
                "ind_zero_days": int((ind == 0).sum().sum())})
    md, tc = load_fredmd()
    md = md.loc[C.START:C.END]
    miss = md.isna().mean().sort_values(ascending=False)
    num.update({"fredmd_n": md.shape[1], "fredmd_months": md.shape[0], "fredmd_last": md.index.max().strftime("%Y-%m"),
                "fredmd_n_missing": int((miss > 0).sum()), "fredmd_max_missing": 100 * float(miss.iloc[0]),
                "fredmd_max_missing_name": miss.index[0], "fredmd_tcodes": int(tc.nunique())})
    fomc = load_fomc_dates()
    num.update({"fomc_n": len(fomc), "fomc_first": fomc["fomc_date"].min().strftime("%Y-%m-%d"),
                "fomc_last": fomc["fomc_date"].max().strftime("%Y-%m-%d")})
    g = load_german_credit()
    num.update({"german_n": len(g), "german_bad": 100 * g["bad"].mean(), "german_attr": g.shape[1] - 1})
    tw = pd.concat(load_fin_tweets())
    num.update({"tweets_n": len(tw), **{f"tweets_{k}": 100 * (tw["label"] == i).mean()
                                        for i, k in enumerate(("bearish", "bullish", "neutral"))}})
    edgar = load_edgar_meta()
    sic = load_ff49_sic()
    edgar["ff49"] = [sic_to_ff49(s, sic) for s in edgar["sic"]]
    num.update({"edgar_n": int(edgar["cik"].notna().sum()), "edgar_sic_n": int(edgar["sic"].notna().sum()),
                "edgar_ff49_n": int(edgar["ff49"].nunique())})
    edgar.to_csv(C.RESULTS_DIR / "lab0_edgar_meta.csv")
    panel = build_panel(5)
    m = split_masks(panel)
    num.update({"panel_series": panel["series"].nunique(), "panel_rows": len(panel),
                "panel_train": int(m["train"].sum()), "panel_val": int(m["val"].sum()), "panel_test": int(m["test"].sum()),
                "panel_ratio": int(m["train"].sum()) / 3510})

    # ------------------------------------------------ figure: coverage
    rows = [(SHORT[t], d["adj_close"].dropna().index.min(), "D1") for t, d in ma.items() if SHORT[t] != "S&P 500"]
    rows += [(VOL_YAHOO.get(c) or VOL_FRED.get(c),
              (download_yahoo(c)["close"] if c in VOL_YAHOO else download_fred(c)).loc[C.START:].dropna().index.min(), "D2")
             for c in list(VOL_YAHOO) + list(VOL_FRED)]
    rows += [("49 industries, 5 factors", ind.index.min(), "D3"), (f"FRED-MD ({md.shape[1]} series)", md.index.min(), "D4"),
             ("FOMC meetings", fomc["fomc_date"].min(), "D6")]
    rows = sorted(rows, key=lambda r: r[1])
    f, ax = P.fig(4.6)
    col = {"D1": P.BLUE, "D2": P.ORANGE, "D3": P.AQUA, "D4": P.VIOLET, "D6": P.MAGENTA}
    for i, (name, start, kind) in enumerate(rows):
        ax.barh(i, (pd.Timestamp(C.END) - start).days, left=start, color=col[kind], height=0.6)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0].replace(" (daily, news-based)", "").replace(" (weekly)", "") for r in rows], fontsize=9)
    ax.invert_yaxis()
    P.shade_splits(ax, label=False)
    for k, c in col.items():
        ax.barh(np.nan, 0, color=c, label=k)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), fontsize=9, ncol=5, frameon=False)
    ax.xaxis_date()
    f.tight_layout(); P.save(f, "lab0_coverage")

    # ------------------------------------------------ figure: cross-asset correlations (training block)
    rets = pd.DataFrame({SHORT[t]: np.log(d["adj_close"]).diff() for t, d in ma.items()})
    rets = rets.loc[:C.TRAIN_END]
    corr = rets.corr(min_periods=250)
    vol = rets.std() * np.sqrt(252) * 100
    f, axs = P.fig(3.6, ncols=2, gridspec_kw={"width_ratios": [1.45, 1]})
    im = axs[0].imshow(corr.to_numpy(), cmap=P.DIV, vmin=-1, vmax=1)
    axs[0].set_xticks(range(len(corr))); axs[0].set_xticklabels(corr.columns, rotation=90, fontsize=9)
    axs[0].set_yticks(range(len(corr))); axs[0].set_yticklabels(corr.index, fontsize=9); axs[0].grid(False)
    f.colorbar(im, ax=axs[0], fraction=0.046, pad=0.02)
    axs[1].barh(range(len(vol)), vol.to_numpy(), color=P.BLUE)
    axs[1].set_yticks(range(len(vol))); axs[1].set_yticklabels(vol.index, fontsize=9); axs[1].invert_yaxis()
    axs[1].set_xlabel("annualized volatility 2000--2014, %")
    f.tight_layout(); P.save(f, "lab0_assets")
    num.update({"corr_spx_tlt": float(corr.loc["S&P 500", "Treasury 20y+"]), "corr_spx_ndx": float(corr.loc["S&P 500", "Nasdaq"]),
                "corr_spx_gold": float(corr.loc["S&P 500", "Gold"]), "vol_max_name": vol.idxmax(), "vol_max": float(vol.max())})

    # ------------------------------------------------ figure: implied-volatility term structure and stress
    p = load_market_panel()
    vix = p["VIXCLS"]
    v9 = download_yahoo("^VIX9D")["close"]; v3 = download_fred("VXVCLS")
    ts = pd.DataFrame({"VIX9D": v9, "VIX": vix, "VIX3M": v3}).loc["2011":C.END].dropna()
    slope = ts["VIX3M"] - ts["VIX"]
    inv_share = 100 * float((slope < 0).mean())
    stress = download_fred("STLFSI4").loc[C.START:C.END]; epu = download_fred("USEPUINDXD").loc[C.START:C.END]
    f, axs = P.fig(4.4, nrows=2, sharex=True)
    axs[0].plot(ts.index, ts["VIX3M"] - ts["VIX9D"], color=P.BLUE, lw=0.6)
    axs[0].axhline(0, color=P.INK2, lw=0.7)
    axs[0].set_ylabel("VIX3M - VIX9D, points")
    axs[1].plot(stress.index, stress, color=P.ORANGE, lw=0.9, label="St. Louis Fed stress index")
    ax2 = axs[1].twinx(); ax2.grid(False)
    ax2.plot(epu.index, epu.rolling(22).mean(), color=P.AQUA, lw=0.7, label="policy uncertainty (22-day mean)")
    axs[1].set_ylabel("stress index"); ax2.set_ylabel("uncertainty index")
    h1, l1 = axs[1].get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    axs[1].legend(h1 + h2, l1 + l2, loc="lower center", bbox_to_anchor=(0.5, 1), ncol=2, fontsize=9)
    P.shade_splits(axs[1], label=False)
    f.tight_layout(); P.save(f, "lab0_stress")
    num.update({"vix_inverted_share": 100 * float(((ts["VIX3M"] - ts["VIX9D"]) < 0).mean()), "vix3m_inverted_share": inv_share,
                "stress_max": float(stress.max()), "stress_max_date": stress.idxmax().strftime("%Y-%m-%d"),
                "epu_max_date": epu.rolling(22).mean().idxmax().strftime("%Y-%m-%d")})
    save_numbers("lab0x", num)
    print(num)


if __name__ == "__main__":
    main()
