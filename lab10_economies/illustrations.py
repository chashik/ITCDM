"""Figures that illustrate the results of stages 1-5 next to their tables in Part 3 (drawn from the saved results and the
panels; nothing is estimated here that the stages do not estimate).

  rates_compare     stage 1: the five-year rates of every country in one unit (% per year): official inflation, the growth
                    of the four products, the policy rate, the yield of government bonds and the return of their holder
  bond_value        stage 1: what 100 placed in government bonds in December 2020 became: interest at the yield against
                    the value of a holder's bonds at market prices (Russia: also the index of the Moscow Exchange)
  gaps              stage 1: growth of each product minus official inflation, by calendar year (p.p.)
  savings           stage 1: real value in December 2025 of 100 placed in December 2020 (cash, policy rate, bonds)
  leadlag           stage 2: correlation of the policy rate with inflation k months earlier or later
  predictability    stage 2: share of a series explained by the others in sample against the skill of a nowcast
  passthrough       stage 3: share of a policy-rate change that has reached the mortgage rate after k months
  crosscountry      stage 3: inflation of the four countries on one axis, with their peaks
  forecast_paths    stage 5: inflation and its forecasts one month ahead, 2023-2025 (short panel)

Run (from code/):  python -m lab10_economies.illustrations   (after growth, relations, patterns, forecasting)
"""
from __future__ import annotations

import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import statsmodels.api as sm

from .common import COUNTRIES, GOODS, LABEL, P, PERIOD, panel, save_figure
from .growth import bond_return
from .paths import RESULTS

YEARS = [2021, 2022, 2023, 2024, 2025]
num = lambda s: s.replace("-", "−")                    # a true minus sign in the labels of the figures
COL = {"official inflation": P.INK, "wheat flour": P.YELLOW, "milk": P.AQUA, "sugar": P.MAGENTA, "petrol": P.ORANGE,
       "policy rate": P.BLUE, "government bond yield": P.VIOLET, "government bond return": P.GREEN}


def read(name: str, **kw) -> pd.DataFrame:
    return pd.read_csv(RESULTS / f"{name}.csv", **kw)


def rates_compare():
    f, axs = P.fig(6.2, nrows=2, ncols=2, sharex=True)
    for ax, c in zip(axs.ravel(), COUNTRIES):
        row = read(f"q1_same_units_{c}", index_col=0).iloc[-1]
        keys = ["official inflation", "wheat flour", "milk", "sugar", "petrol", "policy rate", "government bond yield",
                "government bond return"]
        vals = [float(row[k]) for k in keys]
        y = np.arange(len(keys))[::-1]
        ax.barh(y, vals, color=[COL[k] for k in keys], height=0.7)
        ax.axvline(vals[0], color=P.INK, lw=0.9, ls="--")
        ax.axvline(0, color=P.MUTED, lw=0.6)
        for yi, v in zip(y, vals):
            ax.text(max(v, 0) + 0.3, yi, num(f"{v:.1f}"), va="center", ha="left", fontsize=9)   # right of the bar or of 0
        ax.set_yticks(y, [k.replace("government bond", "gov. bond") for k in keys], fontsize=9)
        ax.set_title(COUNTRIES[c], fontsize=9.5)
    for ax in axs[-1]:
        ax.set_xlabel("% per year, December 2020 – December 2025")
    f.tight_layout()
    save_figure(f, "rates_compare")


def bond_value():
    f, axs = P.fig(5.4, nrows=2, ncols=2, sharex=True)
    for ax, c in zip(axs.ravel(), COUNTRIES):
        p = panel(c, base=True).loc[:PERIOD[1]]
        x = p.index.to_timestamp()
        if c == "BY":                                  # dollar bonds, not traded: held to repayment, value in roubles
            y = p["bond_gov"].interpolate(limit_direction="both")
            accrued = (1 + y.shift(1).fillna(0) / 1200).cumprod()
            held = accrued * p["fx_usd"] / p["fx_usd"].iloc[0]
        else:                                          # traded bonds: coupon and market price (growth.bond_return)
            y = p["bond_market"]
            accrued = (1 + y.shift(1).fillna(0) / 1200).cumprod()
            held = (1 + bond_return(y).fillna(0) / 100).cumprod()
        ax.plot(x, p["cpi_index"], color=P.INK, lw=1.6, ls="--", label="consumer prices")
        ax.plot(x, 100 * accrued, color=P.VIOLET, lw=1.2, label="interest at the yield")
        ax.plot(x, 100 * held, color=P.GREEN, lw=1.8, label="value of the holder's bonds")
        if c == "RU":
            ax.plot(x, 100 * p["bond_index"] / p["bond_index"].iloc[0], color=P.ORANGE, lw=1.2,
                    label="OFZ index of the Moscow Exchange")
        ax.axhline(100, color=P.MUTED, lw=0.6)
        ax.set_title(COUNTRIES[c] + {"BY": " (US dollar bonds, in roubles)", "LT": " (euro-area yield)"}.get(c, ""),
                     fontsize=9.5)
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    for ax in axs[:, 0]:
        ax.set_ylabel("value of 100 placed in Dec 2020")
    h, l = axs[1, 0].get_legend_handles_labels()       # Russia: all four lines
    f.legend(h, l, loc="lower center", ncol=4, fontsize=9, frameon=False)
    f.tight_layout(rect=(0, 0.05, 1, 1))
    save_figure(f, "bond_value")


def gaps():
    f, axs = P.fig(4.6, nrows=2, ncols=2, sharex=True, sharey=True)
    for ax, c in zip(axs.ravel(), COUNTRIES):
        t = read(f"q1_same_units_{c}", index_col=0).iloc[:-1]
        goods = [LABEL[g] for g in GOODS]
        g = np.array([[t.loc[str(y), k] - t.loc[str(y), "official inflation"] for y in YEARS] for k in goods], float)
        im = ax.imshow(np.clip(g, -40, 40), cmap=P.DIV_CELL, vmin=-40, vmax=40, aspect="auto")
        for i in range(len(goods)):
            for j in range(len(YEARS)):
                if np.isfinite(g[i, j]):
                    ax.text(j, i, num(f"{g[i, j]:+.0f}") if round(g[i, j]) else "0", ha="center", va="center", fontsize=9, color=P.INK)
                else:
                    ax.text(j, i, "–", ha="center", va="center", fontsize=9, color=P.INK)
        ax.set_xticks(range(len(YEARS)), YEARS, fontsize=9)
        ax.set_yticks(range(len(goods)), goods, fontsize=9)
        ax.grid(False)
        ax.set_title(COUNTRIES[c], fontsize=9.5)
    f.colorbar(im, ax=axs, shrink=0.8, label="growth minus inflation, p.p. (colour clipped at ±40)")
    save_figure(f, "gaps")


def savings():
    rows = {}
    for c in COUNTRIES:
        s = read(f"q1_savings_{c}", index_col=0)["real value of 100"]
        bond = "government bond in USD, in BYN" if c == "BY" else "government bond, 10 years, holder"
        rows[COUNTRIES[c]] = {"cash": s["cash (no interest)"], "deposit at the policy rate": s["policy rate"],
                              "government bonds": s[bond]}
    t = pd.DataFrame(rows).T
    f, ax = P.fig(2.9)
    x = np.arange(len(t))
    for i, (k, col) in enumerate(zip(t.columns, [P.MUTED, P.BLUE, P.GREEN])):
        ax.bar(x + (i - 1) * 0.26, t[k], width=0.26, color=col, label=k)
        for xi, v in zip(x, t[k]):
            ax.text(xi + (i - 1) * 0.26, v + 1.5, f"{v:.0f}", ha="center", va="bottom", fontsize=9,
                    bbox=dict(fc="white", ec="none", pad=0.15))     # readable over the line at 100
    ax.axhline(100, color=P.INK, lw=0.9, ls="--")
    ax.set_xticks(x, t.index)
    ax.set_ylabel("real value in Dec 2025 of 100 placed in Dec 2020")
    ax.set_ylim(0, 150)
    ax.legend(fontsize=9, ncol=3, loc="upper left", frameon=False)
    f.tight_layout()
    save_figure(f, "savings")


def leadlag():
    f, ax = P.fig(2.9)
    for c, col in zip(COUNTRIES, P.CAT):
        p = panel(c)
        ks = list(range(-12, 13))
        cc = [p["policy_rate"].corr(p["cpi_yoy"].shift(-k)) for k in ks]
        kb = ks[int(np.nanargmax(cc))]
        ax.plot(ks, cc, color=col, lw=1.5, label=COUNTRIES[c])
        ax.plot([kb], [max(cc)], "o", color=col, ms=5)
    ax.axvline(0, color=P.MUTED, lw=0.6)
    ax.set_xlabel("k, months: correlation of the policy rate in month t with inflation in month t + k "
                  "(k < 0: the rate follows inflation)")
    ax.set_ylabel("correlation")
    ax.legend(fontsize=9, frameon=False)
    f.tight_layout()
    save_figure(f, "leadlag")


def predictability():
    pr = read("q2_predictability")
    f, axs = P.fig(5.6, nrows=2, ncols=2, sharex=True)
    for ax, c in zip(axs.ravel(), COUNTRIES):
        q = pr[pr["country"] == c].set_index("target")
        y = np.arange(len(q))[::-1]
        ax.barh(y + 0.18, q["R2 from the others (in sample)"], height=0.34, color=P.MUTED, label="$R^2$ in sample")
        ax.barh(y - 0.18, q["skill with own last value"].clip(-1, 1), height=0.34, color=P.BLUE,
                label="nowcast skill out of sample (clipped at -1)")
        ax.axvline(0, color=P.INK, lw=0.7)
        ax.set_yticks(y, q.index, fontsize=9)
        ax.set_title(COUNTRIES[c], fontsize=9.5)
        ax.set_xlim(-1.05, 1.05)
    h, l = axs[0, 0].get_legend_handles_labels()
    f.legend(h, l, loc="lower center", ncol=2, fontsize=9, frameon=False)
    f.tight_layout(rect=(0, 0.05, 1, 1))
    save_figure(f, "predictability")


def passthrough():
    f, ax = P.fig(2.8)
    for c, col in zip(COUNTRIES, P.CAT):
        p = panel(c)
        d = pd.DataFrame({"dm": p["mortgage_rate"].diff()} | {f"dp{k}": p["policy_rate"].diff().shift(k) for k in range(7)}).dropna()
        fit = sm.OLS(d["dm"], sm.add_constant(d.drop(columns="dm"))).fit()
        cum = np.cumsum([fit.params[f"dp{k}"] for k in range(7)])
        ax.plot(range(7), cum, "o-", color=col, lw=1.5, ms=4, label=COUNTRIES[c])
    ax.axhline(1, color=P.INK, lw=0.8, ls="--")
    ax.axhline(0, color=P.MUTED, lw=0.6)
    ax.set_xlabel("months after a change of the policy rate")
    ax.set_ylabel("share reached the mortgage rate")
    ax.legend(fontsize=9, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1), ncol=4)
    f.tight_layout()
    save_figure(f, "passthrough")


def crosscountry():
    f, ax = P.fig(2.9)
    for c, col in zip(COUNTRIES, P.CAT):
        s = panel(c)["cpi_yoy"]
        k = s.idxmax()
        ax.plot(s.index.to_timestamp(), s, color=col, lw=1.6, label=f"{COUNTRIES[c]} (peak {s.max():.1f} %, {k})")
        ax.plot([k.to_timestamp()], [s.max()], "o", color=col, ms=5)
    ax.axhline(0, color=P.MUTED, lw=0.6)
    ax.set_ylabel("official inflation, % over 12 months")
    ax.legend(fontsize=9, frameon=False, ncol=2, loc="upper right")
    ax.set_ylim(top=30)
    f.tight_layout()
    save_figure(f, "crosscountry")


def forecast_paths():
    t = read("q4_paths")
    t = t[t["panel"] == "short 2021-2025"]
    models = [("random walk", P.MUTED, "--", "no change"), ("AR (own lags)", P.BLUE, "-", "autoregression"),
              ("ridge", P.GREEN, "-", "ridge"), ("random forest", P.ORANGE, "-", "random forest")]
    f, axs = P.fig(5.4, nrows=2, ncols=2, sharex=True)
    for ax, c in zip(axs.ravel(), COUNTRIES):
        g = t[t["country"] == c]
        tgt = (pd.PeriodIndex(g[g.model == "random walk"]["origin"], freq="M") + 1).to_timestamp()
        ax.plot(tgt, g[g.model == "random walk"]["actual"], color=P.INK, lw=2.0, label="inflation (actual)")
        for m, col, ls, lab in models:
            gm = g[g.model == m]
            ax.plot((pd.PeriodIndex(gm["origin"], freq="M") + 1).to_timestamp(), gm["forecast"], color=col, ls=ls, lw=1.1,
                    label=lab)
        ax.set_title(COUNTRIES[c], fontsize=9.5)
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    for ax in axs[:, 0]:
        ax.set_ylabel("% over 12 months")
    h, l = axs[0, 0].get_legend_handles_labels()
    f.legend(h, l, loc="lower center", ncol=5, fontsize=9, frameon=False)
    f.tight_layout(rect=(0, 0.05, 1, 1))
    save_figure(f, "forecast_paths")


def main():
    for fn in (rates_compare, bond_value, gaps, savings, leadlag, predictability, passthrough, crosscountry,
               forecast_paths):
        fn()
        print("figure", fn.__name__)


if __name__ == "__main__":
    main()
