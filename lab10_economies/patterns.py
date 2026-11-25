"""Question 3: derived data that can be mined from the series (patterns, regularities, conclusions).

Outputs (results/lab10; figures paper/figures/lab10_*.pdf):
  q3_price_setting.csv   how prices change month to month: share of unchanged months, jumps above 5 %, the largest jump
  q3_episode.csv         anatomy of the inflation episode: peak, months above 10 %, first rate hike, rate peak, first cut,
                         first month with a positive real policy rate, lags between them
  q3_passthrough.csv     share of a policy-rate change that reaches the mortgage rate after 0-6 months (distributed-lag
                         regression of monthly changes) and the average mortgage - policy spread
  q3_regimes.csv         inflation regimes: segments of a piecewise-constant mean of inflation (least squares, number of
                         segments by BIC)
  q3_regimes_uncapped.csv  number of segments chosen by BIC with the cap of six and up to 15 segments
  q3_seasonality.csv     average month-on-month change by calendar month (2021-2025)
  q3_crosscountry.csv    co-movement of the inflation of every pair of countries, and which country leads
  q3_price_setting.png, q3_regimes.png
Run (from code/):  python -m lab10_economies.patterns
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm

from .common import COUNTRIES, GOODS, LABEL, P, panel, save_figure, save_table


def mom(p: pd.DataFrame) -> pd.DataFrame:
    m = pd.DataFrame({LABEL[g]: 100 * p[f"price_{g}"].pct_change() for g in GOODS})
    m["consumer prices"] = p["cpi_mom"]
    return m.iloc[1:]


def price_setting(p: pd.DataFrame) -> pd.DataFrame:
    rows = {}
    for g, s in mom(p).items():
        s = s.dropna()
        if len(s) < 24:                                      # Russian sugar and petrol: December values only
            continue
        k = s.abs().idxmax()
        rows[g] = {"unchanged months (|change| < 0.1 %), %": 100 * (s.abs() < 0.1).mean(),
                   "months with a jump > 5 %": int((s.abs() > 5).sum()),
                   "largest monthly change, %": s[k], "in": str(k),
                   "share of 5-year growth in the 3 largest months, %":
                       100 * np.log1p(s.nlargest(3) / 100).sum() / np.log1p(s / 100).sum()}
    return pd.DataFrame(rows).T


def episode(p: pd.DataFrame) -> dict:
    cpi, pol, real = p["cpi_yoy"], p["policy_rate"], p["real_policy_rate"]
    peak = cpi.idxmax()
    hikes = pol.diff() > 0
    first_hike = pol.index[hikes][0] if hikes.any() else None
    pol_peak = pol.idxmax()
    cuts = (pol.diff() < 0) & (pol.index > pol_peak)
    first_cut = pol.index[cuts][0] if cuts.any() else None
    pos = real[(real > 0) & (real.index > peak)]
    below5 = cpi[(cpi < 5) & (cpi.index > peak)]
    months = lambda a, b: (b - a).n if a is not None and b is not None else np.nan
    return {"inflation peak, %": cpi.max(), "peak month": str(peak), "months above 10 %": int((cpi > 10).sum()),
            "first month below 5 % after the peak": str(below5.index[0]) if len(below5) else "–",
            "first rate hike": str(first_hike), "policy-rate peak, %": pol.max(), "rate peak reached": str(pol_peak),
            "first cut": str(first_cut), "real policy rate positive from": str(pos.index[0]) if len(pos) else "–",
            "months from first hike to inflation peak": months(first_hike, peak),
            "months from inflation peak to first cut": months(peak, first_cut),
            "lowest real policy rate, p.p.": real.min()}


def passthrough(p: pd.DataFrame, lags: int = 6) -> dict:
    d = pd.DataFrame({"dm": p["mortgage_rate"].diff()} | {f"dp{k}": p["policy_rate"].diff().shift(k) for k in range(lags + 1)}).dropna()
    fit = sm.OLS(d["dm"], sm.add_constant(d.drop(columns="dm"))).fit(cov_type="HAC", cov_kwds={"maxlags": 3})
    cum = np.cumsum([fit.params[f"dp{k}"] for k in range(lags + 1)])
    spread = p["mortgage_rate"] - p["policy_rate"]
    return {"pass-through in the same month": cum[0], "after 3 months": cum[3], "after 6 months": cum[6],
            "R2": fit.rsquared, "mortgage - policy spread, mean p.p.": spread.mean(), "spread s.d.": spread.std()}


def segments(y: np.ndarray, kmax: int = 6, minlen: int = 4) -> list[int]:
    """Exact least-squares segmentation into k segments of constant mean (dynamic programming); k chosen by BIC."""
    n = len(y)
    cs, cs2 = np.r_[0, np.cumsum(y)], np.r_[0, np.cumsum(y ** 2)]
    cost = lambda i, j: cs2[j] - cs2[i] - (cs[j] - cs[i]) ** 2 / (j - i)
    best, arg = {(1, j): cost(0, j) for j in range(minlen, n + 1)}, {}
    for k in range(2, kmax + 1):
        for j in range(k * minlen, n + 1):
            c = [(best[(k - 1, i)] + cost(i, j), i) for i in range((k - 1) * minlen, j - minlen + 1) if (k - 1, i) in best]
            if c:
                best[(k, j)], arg[(k, j)] = min(c)
    bic = {k: n * np.log(best[(k, n)] / n) + 2 * k * np.log(n) for k in range(1, kmax + 1) if (k, n) in best}
    k = min(bic, key=bic.get)
    cuts, j = [], n
    for kk in range(k, 1, -1):
        j = arg[(kk, j)]
        cuts.append(j)
    return sorted(cuts)


def regimes(p: pd.DataFrame, country: str) -> pd.DataFrame:
    y = p["cpi_yoy"].to_numpy()
    edges = [0] + segments(y) + [len(y)]
    return pd.DataFrame([{"country": country, "from": str(p.index[a]), "to": str(p.index[b - 1]),
                          "mean inflation, %": y[a:b].mean(), "mean policy rate, %": p["policy_rate"].iloc[a:b].mean(),
                          "mean real policy rate, p.p.": p["real_policy_rate"].iloc[a:b].mean()}
                         for a, b in zip(edges[:-1], edges[1:])])


def seasonality(p: pd.DataFrame) -> pd.DataFrame:
    m = mom(p)
    t = m.groupby(m.index.month).mean().T
    t.columns = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    return t


def main():
    panels = {c: panel(c) for c in COUNTRIES}
    ps, ep, pt, rg, se = {}, {}, {}, [], []
    for c, p in panels.items():
        ps[c] = price_setting(p)
        save_table(ps[c], f"q3_price_setting_{c}")
        ep[c] = episode(p)
        pt[c] = passthrough(p)
        rg.append(regimes(p, c))
        s = seasonality(p)
        save_table(s, f"q3_seasonality_{c}")
    ept, ptt, rgt = pd.DataFrame(ep), pd.DataFrame(pt).T, pd.concat(rg).set_index("country")
    save_table(ept, "q3_episode"); save_table(ptt, "q3_passthrough", 3); save_table(rgt, "q3_regimes")
    # BIC assumes independent errors; inflation over 12 months is strongly autocorrelated, and without the cap of six
    # segments BIC would choose more (the cap is a modelling decision, stated in the text)
    unc = pd.DataFrame({"segments, BIC up to 6": [len(segments(p["cpi_yoy"].to_numpy())) + 1 for p in panels.values()],
                        "segments, BIC up to 15": [len(segments(p["cpi_yoy"].to_numpy(), kmax=15)) + 1 for p in panels.values()],
                        "lag-1 autocorrelation of inflation": [p["cpi_yoy"].autocorr(1) for p in panels.values()]},
                       index=pd.Index(list(panels), name="country"))
    save_table(unc, "q3_regimes_uncapped", 3)
    rows = []
    codes = list(panels)
    for i, a in enumerate(codes):                              # every pair: k > 0 means the first country leads
        for b in codes[i + 1:]:
            x, y = panels[a]["cpi_yoy"], panels[b]["cpi_yoy"]
            cc = {k: x.corr(y.shift(-k)) for k in range(-12, 13)}
            kbest = max(cc, key=cc.get)
            rows.append({"pair": f"{a}-{b}", "correlation, same month": cc[0],
                         "lag with the highest correlation (k > 0: first leads by k months)": kbest,
                         "correlation at that lag": cc[kbest]})
    cross = pd.DataFrame(rows).set_index("pair")
    save_table(cross, "q3_crosscountry", 3)

    f, axs = P.fig(5.6, nrows=2, ncols=2, sharey=True)         # monthly price changes: steps against continuous moves
    for ax, (c, p) in zip(axs.ravel(), panels.items()):
        m = mom(p)
        for i, (g, s) in enumerate(m.items()):
            x = np.full(len(s), i) + np.random.default_rng(i).uniform(-0.25, 0.25, len(s))
            ax.scatter(x, s.clip(-15, 30), s=7, color=P.CAT[i], alpha=0.7, lw=0)
        ax.set_xticks(range(m.shape[1]), m.columns, rotation=30, ha="right", fontsize=9)
        ax.axhline(0, color=P.MUTED, lw=0.6)
        ax.set_title(COUNTRIES[c], fontsize=9.5)
    for ax in axs[:, 0]:
        ax.set_ylabel("monthly change, % (clipped at 30)")
    f.tight_layout()
    save_figure(f, "q3_price_setting")

    f, axs = P.fig(5.6, nrows=2, ncols=2, sharey=True)         # inflation regimes and the policy rate
    for ax, (c, p) in zip(axs.ravel(), panels.items()):
        x = p.index.to_timestamp()
        ax.plot(x, p["cpi_yoy"], color=P.INK, lw=1.4, label="inflation, y/y")
        ax.plot(x, p["policy_rate"], color=P.BLUE, lw=1.2, label="policy rate")
        for _, r in rgt.loc[[c]].iterrows():
            a, b = pd.Period(r["from"], "M").to_timestamp(), (pd.Period(r["to"], "M") + 1).to_timestamp()
            ax.hlines(r["mean inflation, %"], a, b, color=P.ORANGE, lw=2.2, label="inflation regime (mean)")
        ax.set_title(COUNTRIES[c], fontsize=9.5)
    h, l = axs[0, 0].get_legend_handles_labels()
    axs[0, 0].legend(h[:3], l[:3], fontsize=9)
    for ax in axs[:, 0]:
        ax.set_ylabel("% per year")
    f.tight_layout()
    save_figure(f, "q3_regimes")
    pd.set_option("display.width", 220)
    for c in COUNTRIES:
        print(c); print(ps[c].round(2).to_string(), "\n"); print(seasonality(panels[c]).round(2).to_string(), "\n")
    print(ept.to_string(), "\n"); print(ptt.round(3).to_string(), "\n"); print(rgt.round(2).to_string(), "\n")
    print(cross.round(3).to_string())


if __name__ == "__main__":
    main()
