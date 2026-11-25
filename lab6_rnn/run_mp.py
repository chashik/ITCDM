"""Lab 6 (iteration 2). Matrix profile and Isolation Forest.

* Matrix profile of the S&P 500 log price (window of 30 days): motif pairs and
  discords over 2000 - June 2026;
* out-of-sample discords: every window of the test block is compared only with the
  windows of 2000-2018 (an AB-join), so no future information is used;
* Isolation Forest on 20-day windows of the eight channels of Lab 6, threshold from
  the validation block, compared with the rule |r| > 4 sigma used for the autoencoder.

Run:  python -m lab6_rnn.run_mp
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import stumpy
from sklearn.ensemble import IsolationForest

from dmfin import config as C
from dmfin import plotting as P
from dmfin.reporting import save_numbers, save_table

from .common import lab6_frame
from .nets import INPUTS, SeqData
from .run_anomaly import episodes

M = 30


def top_k(profile, k, excl, largest):
    """Indices of the k smallest (or largest) profile values, at least ``excl`` apart."""
    order = np.argsort(profile)[::-1] if largest else np.argsort(profile)
    out = []
    for i in order:
        if np.isfinite(profile[i]) and all(abs(i - j) >= excl for j in out):
            out.append(int(i))
        if len(out) == k:
            break
    return out


def main():
    num = {"m": M}
    df, panel = lab6_frame()
    logp = np.log(panel["adj_close"]).dropna()
    x = logp.to_numpy(float); d = logp.index
    mp = stumpy.stump(x, M)
    prof = np.asarray(mp[:, 0], dtype=float)
    motifs = []                                                    # distinct pairs: skip the partner of a listed window
    for i in np.argsort(prof):
        j = int(mp[i, 1])
        if all(abs(i - k) >= M and abs(j - k) >= M for k in motifs + [int(mp[k, 1]) for k in motifs]):
            motifs.append(int(i))
        if len(motifs) == 3:
            break
    discords = top_k(prof, 5, M, largest=True)
    rows = [("motif", d[i].strftime("%Y-%m-%d"), d[int(mp[i, 1])].strftime("%Y-%m-%d"), float(prof[i])) for i in motifs]
    rows += [("discord", d[i].strftime("%Y-%m-%d"), "--", float(prof[i])) for i in discords]
    save_table("lab6_mp", pd.DataFrame(rows, columns=["Kind", "Window start", "Nearest window", "Distance"]).set_index("Kind"),
               nd={"Distance": 2}, col_format="lccc")
    num.update({"n_windows": len(prof), "disc1": d[discords[0]].strftime("%Y-%m-%d"),
                "disc2": d[discords[1]].strftime("%Y-%m-%d"), "motif_dist": float(prof[motifs[0]]),
                "prof_median": float(np.nanmedian(prof))})

    # ---------------- out-of-sample discords: test windows against 2000-2018 only (AB-join)
    split = np.searchsorted(d, pd.Timestamp(C.VAL_END), side="right")
    hist, test = x[:split], x[split - M + 1:]
    ab = stumpy.stump(test, M, hist, ignore_trivial=False)
    pab = np.asarray(ab[:, 0], dtype=float)
    dt = d[split - M + 1:][:len(pab)]
    end_dates = d[split:][:len(pab)]                                   # a window is known at its last day
    top_ab = top_k(pab, 5, M, largest=True)
    num.update({f"ab_disc{k + 1}": end_dates[i].strftime("%Y-%m-%d") for k, i in enumerate(top_ab[:3])})
    num["ab_q99"] = float(np.quantile(pab, 0.99))
    f, axs = P.fig(4.2, nrows=3, sharex=False, gridspec_kw={"height_ratios": [1.2, 0.8, 0.9]})
    axs[0].plot(d, np.exp(x), color=P.INK2, lw=0.6); axs[0].set_yscale("log"); axs[0].set_ylabel("S\\&P 500")
    for i in discords:
        axs[0].plot(d[i:i + M], np.exp(x[i:i + M]), color=P.RED, lw=1.6)
    for i in motifs[:1]:
        j = int(mp[i, 1])
        for k in (i, j):
            axs[0].plot(d[k:k + M], np.exp(x[k:k + M]), color=P.BLUE, lw=1.6)
    axs[1].plot(d[:len(prof)], prof, color=P.BLUE, lw=0.5); axs[1].set_ylabel("matrix profile")
    axs[2].plot(end_dates, pab, color=P.ORANGE, lw=0.6); axs[2].set_ylabel("distance to\n2000--2018", fontsize=8)
    for i in top_ab[:3]:
        axs[2].axvline(end_dates[i], color=P.RED, ls=":", lw=0.8)
    f.tight_layout(); P.save(f, "lab6_mp")

    # ---------------- Isolation Forest on 20-day windows
    sd = SeqData(df, "y_logrv_5", lookback=20, horizon=0)
    feat = lambda X: np.concatenate([X.mean(1), X.std(1), X.min(1), X.max(1), X[:, -1]], axis=1)
    iso = IsolationForest(n_estimators=500, random_state=C.SEED).fit(feat(sd["train"].X))
    s_val = -iso.score_samples(feat(sd["val"].X)); s_te = -iso.score_samples(feat(sd["test"].X))
    thr = float(np.quantile(s_val, 0.99))
    flag = s_te > thr
    dte = sd["test"].dates
    eps = episodes(dte[flag])
    r = df["ret"]
    z = (r / r.rolling(252).std().shift(1)).abs()
    base_days = z.loc[dte].index[z.loc[dte] > 4]
    near = lambda day, S: np.any(np.abs((S - day).days) <= 5) if len(S) else False
    num.update({"if_n_flag": int(flag.sum()), "if_share": 100 * flag.mean(), "if_episodes": len(eps),
                "if_base_caught": int(sum(near(day, dte[flag]) for day in base_days)), "if_n_base": len(base_days),
                "if_episode_list": "; ".join(f"{a:%Y-%m-%d} -- {b:%Y-%m-%d}" for a, b in eps[:6])})
    f, ax = P.fig(2.6)
    ax.plot(dte, s_te, color=P.BLUE, lw=0.6); ax.axhline(thr, color=P.RED, ls="--", lw=0.9, label="99 % of validation")
    ax.scatter(dte[flag], s_te[flag], s=6, color=P.RED, zorder=3)
    ax.set_ylabel("isolation score"); ax.legend(fontsize=7.5)
    P.save(f, "lab6_iforest")
    save_numbers("lab6p", num)
    print(num)


if __name__ == "__main__":
    main()
