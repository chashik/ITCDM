"""Lab 5. Data-set analysis with fuzzy-logic tools.

  7.1 fuzzy c-means on the Lab 1 stock descriptors;
  7.2 Mamdani fuzzy inference: an interpretable market-stress index;
  7.3 Wang-Mendel fuzzy rules for credit default (Lab 3 data).

Run:  python -m lab5_fuzzy.run
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import adjusted_rand_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from dmfin import config as C
from dmfin import plotting as P
from dmfin.data import load_credit, load_market_panel, load_stock_prices
from dmfin.features import add_targets, market_features
from dmfin.reporting import save_numbers, save_table
from lab3_classification.model import prepare
from lab1_clustering.model import stock_features

from .fuzzy import (Mamdani, Partition, WangMendel, fcm, partition_coefficient, partition_entropy,
                    trapmf, trimf, xie_beni)


def part_fcm(num):
    px = load_stock_prices()
    panel = load_market_panel()
    R = np.log(px).diff().iloc[1:].loc[C.STOCKS_START:C.STOCKS_SPLIT]
    feats = stock_features(R, np.log(panel["adj_close"]).diff().loc[R.index], panel["DGS10"].diff().loc[R.index],
                           panel["DCOILWTICO"].pct_change(fill_method=None).clip(-0.5, 0.5).loc[R.index])
    X = StandardScaler().fit_transform(feats)
    rows = []
    for c in range(2, 11):
        U, V, J, it = fcm(X, c, m=2.0, seed=C.SEED)
        rows.append((c, J, partition_coefficient(U), partition_entropy(U), xie_beni(X, U, V), 1 / c))
    vi = pd.DataFrame(rows, columns=["c", "J", "PC", "PE", "XB", "1/c"]).set_index("c")
    vt = vi.copy()
    sci = lambda v: (lambda m, e: f"${m}\\cdot 10^{{{int(e)}}}$")(*f"{v:.1e}".split("e"))
    vt["XB"] = [f"{v:.3f}" if v < 100 else sci(v) for v in vi["XB"]]
    save_table("lab5_fcm_validity", vt, nd={"J": 1, "PC": 3, "PE": 3, "1/c": 3})
    c_xb = int(vi["XB"].idxmin())
    num.update({"fcm_c_xb": c_xb, "fcm_pc_c2": vi.loc[2, "PC"], "fcm_pc_c6": vi.loc[6, "PC"]})

    km = KMeans(6, n_init=100, random_state=C.SEED).fit(X).labels_
    mrows = []
    for m in (1.1, 1.5, 2.0, 2.5, 3.0):
        U, V, J, it = fcm(X, 6, m=m, seed=C.SEED)
        mrows.append((m, U.max(1).mean(), (U.max(1) < 0.5).mean() * 100, adjusted_rand_score(km, U.argmax(1)), it))
    mt = pd.DataFrame(mrows, columns=["$m$", "Mean max. membership", "Ambiguous stocks, %", "ARI with k-means",
                                      "Iterations"]).set_index("$m$")
    save_table("lab5_fcm_m", mt, nd={"Mean max. membership": 3, "Ambiguous stocks, %": 1, "ARI with k-means": 3,
                                     "Iterations": 0})
    num.update({"fcm_ari_m11": mt.iloc[0, 2], "fcm_ari_m2": mt.iloc[2, 2], "fcm_maxmu_m2": mt.iloc[2, 0],
                "fcm_amb_m2": mt.iloc[2, 1], "fcm_maxmu_m3": mt.iloc[4, 0]})

    U, V, _, _ = fcm(X, 6, m=2.0, seed=C.SEED)
    # order clusters by mean beta, as in Lab 1
    beta = feats["beta_mkt"].to_numpy()
    order = np.argsort([(U[:, j] * beta).sum() / U[:, j].sum() for j in range(6)])
    U = U[:, order]
    amb = np.argsort(U.max(1))[:18]
    f, ax = P.fig(3.0)
    left = np.zeros(len(amb))
    for j in range(6):
        ax.barh(range(len(amb)), U[amb, j], left=left, color=P.CAT[j], height=0.75, label=f"F{j+1}",
                edgecolor="white", linewidth=0.6)
        left += U[amb, j]
    ax.set_yticks(range(len(amb))); ax.set_yticklabels(feats.index[amb], fontsize=7.5)
    ax.set_xlabel("membership degree $u_{ij}$"); ax.set_xlim(0, 1)
    ax.legend(ncol=6, fontsize=7.5, loc="lower center", bbox_to_anchor=(0.5, 1.0)); ax.grid(axis="y", visible=False)
    P.save(f, "lab5_fcm_members")
    f, axs = P.fig(2.5, ncols=2)
    axs[0].plot(vi.index, vi["PC"], "o-", ms=3.5, color=P.BLUE, label="partition coefficient")
    axs[0].plot(vi.index, vi["PE"] / np.log(vi.index), "s-", ms=3.5, color=P.ORANGE, label="normalized entropy")
    axs[0].plot(vi.index, vi["1/c"], ":", color=P.MUTED, label="$1/c$")
    axs[0].set_xlabel("number of clusters $c$"); axs[0].legend(fontsize=7.5)
    axs[1].plot(vi.index, vi["XB"], "^-", ms=3.5, color=P.AQUA)
    axs[1].set_yscale("log"); axs[1].set_xlabel("number of clusters $c$"); axs[1].set_ylabel("Xie-Beni index (log scale)")
    f.tight_layout(); P.save(f, "lab5_fcm_validity")
    num["fcm_amb_names"] = ", ".join(feats.index[amb[:6]])


def stress_system(tr):
    q = lambda c, p: float(tr[c].quantile(p))
    inputs = {"vix": Partition(q("vix", 0.1), q("vix", 0.5), q("vix", 0.9)),
              "spread": Partition(q("credit_baa_aaa", 0.1), q("credit_baa_aaa", 0.5), q("credit_baa_aaa", 0.9),
                                  names=("narrow", "normal", "wide")),
              "mom": Partition(q("ret_22", 0.1), q("ret_22", 0.5), q("ret_22", 0.9),
                               names=("falling", "flat", "rising"))}
    output = {"calm": lambda u: trapmf(u, -1, 0, 15, 35), "elevated": lambda u: trimf(u, 20, 40, 60),
              "stressed": lambda u: trimf(u, 45, 65, 85), "crisis": lambda u: trapmf(u, 70, 90, 100, 101)}
    rules = [({"vix": "low", "spread": "narrow"}, "calm"), ({"vix": "low", "mom": "rising"}, "calm"),
             ({"vix": "low", "spread": "normal"}, "calm"), ({"vix": "low", "spread": "wide"}, "elevated"),
             ({"vix": "low", "mom": "falling"}, "elevated"),
             ({"vix": "medium", "spread": "narrow"}, "elevated"), ({"vix": "medium", "spread": "normal"}, "elevated"),
             ({"vix": "medium", "mom": "flat"}, "elevated"), ({"vix": "medium", "spread": "wide"}, "stressed"),
             ({"vix": "medium", "mom": "falling"}, "stressed"),
             ({"vix": "high"}, "stressed"), ({"vix": "high", "mom": "rising"}, "stressed"),
             ({"vix": "high", "mom": "falling"}, "crisis"), ({"vix": "high", "spread": "wide"}, "crisis"),
             ({"spread": "wide", "mom": "falling"}, "crisis")]
    return Mamdani(inputs, output, np.linspace(0, 100, 501), rules), rules


def part_mamdani(num):
    feats = add_targets(market_features(load_market_panel()), horizons=(5, 22))
    d = feats[["vix", "credit_baa_aaa", "ret_22", "y_rv_5", "y_rv_22", "recession"]].dropna()
    tr = d.loc[:C.TRAIN_END]
    fis, rules = stress_system(tr)
    stress = pd.Series(fis({"vix": d["vix"], "spread": d["credit_baa_aaa"], "mom": d["ret_22"]}), index=d.index)
    q80 = tr["y_rv_5"].quantile(0.8)
    te = d.loc[pd.Timestamp(C.VAL_END) + pd.Timedelta(days=1):]
    st_te = stress.loc[te.index]
    lab = (te["y_rv_5"] > q80).astype(int)
    trva = d.loc[:C.VAL_END].iloc[:-5]              # purge: the 5-day targets of the last days reach into the test block
    lr = make_pipeline(StandardScaler(), LogisticRegression()).fit(trva[["vix", "credit_baa_aaa", "ret_22"]],
                                                                 (trva["y_rv_5"] > q80).astype(int))
    p_lr = lr.predict_proba(te[["vix", "credit_baa_aaa", "ret_22"]])[:, 1]
    num.update({"n_rules": len(rules), "stress_auc": roc_auc_score(lab, st_te), "vix_auc": roc_auc_score(lab, te["vix"]),
                "lr3_auc": roc_auc_score(lab, p_lr),
                "stress_spear22": stats.spearmanr(st_te, te["y_rv_22"]).statistic,
                "vix_spear22": stats.spearmanr(te["vix"], te["y_rv_22"]).statistic,
                "stress_max": float(stress.max()), "stress_max_date": stress.idxmax().strftime("%Y-%m-%d"),
                "stress_mean_rec": float(stress[d["recession"] == 1].mean()),
                "stress_mean_norec": float(stress[d["recession"] == 0].mean()),
                "vix_lo": fis.inputs["vix"].lo, "vix_mid": fis.inputs["vix"].mid, "vix_hi": fis.inputs["vix"].hi,
                "sp_lo": fis.inputs["spread"].lo, "sp_mid": fis.inputs["spread"].mid, "sp_hi": fis.inputs["spread"].hi,
                "mom_lo": 100 * fis.inputs["mom"].lo, "mom_mid": 100 * fis.inputs["mom"].mid,
                "mom_hi": 100 * fis.inputs["mom"].hi})
    # stress level bands vs future volatility (test period)
    bands = pd.cut(st_te, [0, 35, 55, 75, 100], labels=["calm", "elevated", "stressed", "crisis"])
    bt = te.groupby(bands, observed=False).agg(days=("y_rv_22", "size"), rv=("y_rv_22", "mean"),
                                               p_stress=("y_rv_5", lambda s: (s > q80).mean()))
    bt["rv"] *= 100; bt["p_stress"] *= 100
    bt.index.name = "Stress band"
    save_table("lab5_bands", bt.rename(columns={"days": "Test days", "rv": "Next-22-day vol., %",
                                                "p_stress": "Stress-week share, %"}),
               nd={"Test days": 0, "Next-22-day vol., %": 1, "Stress-week share, %": 1})

    f, axs = P.fig(4.2, nrows=2, ncols=2)
    for ax, (v, lab_, scale) in zip(axs.ravel()[:3], [("vix", "VIX", 1), ("spread", "Baa - Aaa spread, p.p.", 1),
                                                       ("mom", "22-day return, %", 100)]):
        p = fis.inputs[v]
        lo = p.lo - (p.hi - p.lo) * 0.4; hi = p.hi + (p.hi - p.lo) * 0.4
        x = np.linspace(lo, hi, 400)
        for j, (k, mu) in enumerate(p(x).items()):
            ax.plot(x * scale, mu, color=P.CAT[j], lw=1.3, label=k)
        ax.set_xlabel(lab_); ax.legend(fontsize=7, loc="center right"); ax.set_ylim(0, 1.08)
    ax = axs[1, 1]
    for j, (k, mu) in enumerate(fis.out_mf.items()):
        ax.plot(fis.universe, mu, color=P.CAT[j], lw=1.3, label=k)
    ax.set_xlabel("stress index"); ax.legend(fontsize=7, loc="center right"); ax.set_ylim(0, 1.08)
    for a in axs[:, 0]:
        a.set_ylabel("membership")
    f.tight_layout(); P.save(f, "lab5_mf")

    vg = np.linspace(10, 60, 90); mg = np.linspace(-0.25, 0.12, 90)
    VV, MM = np.meshgrid(vg, mg)
    Z = fis({"vix": VV.ravel(), "spread": np.full(VV.size, tr["credit_baa_aaa"].median()), "mom": MM.ravel()}).reshape(VV.shape)
    f, ax = P.fig(3.0, w=4.6)
    cs = ax.contourf(VV, 100 * MM, Z, levels=16, cmap=P.SEQ)
    f.colorbar(cs, ax=ax, label="stress index"); ax.set_xlabel("VIX"); ax.set_ylabel("22-day return, %"); ax.grid(False)
    P.save(f, "lab5_surface")

    f, ax = P.fig(2.6)
    P.shade_recessions(ax, d["recession"])
    ax.plot(stress.index, stress.values, color=P.BLUE, lw=0.6)
    ax.set_ylabel("fuzzy stress index"); ax.set_ylim(0, 100)
    P.save(f, "lab5_stress_series")


def part_wang_mendel(num):
    X, y = prepare(load_credit())
    Xtrva, Xte, ytrva, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=C.SEED)
    Xtr, Xva, ytr, yva = train_test_split(Xtrva, ytrva, test_size=0.25, stratify=ytrva, random_state=C.SEED)
    cols = ["pay_1", "utilisation", "log_limit", "pay_ratio"]
    q = lambda c, p: float(Xtr[c].quantile(p))
    parts = {"pay_1": Partition(-1.0, 0.0, 2.0, names=("paid", "revolving", "late")),
             "utilisation": Partition(q("utilisation", 0.1), q("utilisation", 0.5), q("utilisation", 0.9)),
             "log_limit": Partition(q("log_limit", 0.1), q("log_limit", 0.5), q("log_limit", 0.9)),
             "pay_ratio": Partition(q("pay_ratio", 0.1), q("pay_ratio", 0.5), q("pay_ratio", 0.9))}
    wm = WangMendel(parts).fit(Xtr[cols], ytr)
    p_te = wm.predict_proba(Xte[cols])[:, 1]
    lr4 = make_pipeline(StandardScaler(), LogisticRegression()).fit(Xtr[cols], ytr)
    lrall = make_pipeline(StandardScaler(), LogisticRegression(C=10, max_iter=2000)).fit(Xtr, ytr)
    num.update({"wm_rules": len(wm.rules_), "wm_auc": roc_auc_score(yte, p_te),
                "lr4_auc": roc_auc_score(yte, lr4.predict_proba(Xte[cols])[:, 1]),
                "lrall_auc": roc_auc_score(yte, lrall.predict_proba(Xte)[:, 1]),
                "wm_auc_val": roc_auc_score(yva, wm.predict_proba(Xva[cols])[:, 1])})
    rt = wm.rule_table(prior=ytr.mean()).head(10)
    rt["support"] = rt["support"].round(0).astype(int)
    rt = rt.rename(columns={"pay_1": "PAY\\_1", "utilisation": "Utilization", "log_limit": "Limit",
                            "pay_ratio": "Pay ratio", "support": "Support", "conf_default": "P(default)", "lift": "Lift"})
    save_table("lab5_rules", rt.set_index("PAY\\_1"), nd={"Support": 0, "P(default)": 3, "Lift": 2})
    top = wm.rule_table(prior=ytr.mean())
    top = top[top["support"] > 50].sort_values("conf_default")
    num.update({"wm_rule_low": float(top["conf_default"].iloc[0]), "wm_rule_high": float(top["conf_default"].iloc[-1]),
                "prior": 100 * ytr.mean()})


def main():
    num = {}
    part_fcm(num)
    part_mamdani(num)
    part_wang_mendel(num)
    save_numbers("lab5", num)
    print(num)


if __name__ == "__main__":
    main()
