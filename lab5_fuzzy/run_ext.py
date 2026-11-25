"""Lab 5 (iteration 2). Crisp association rules (Apriori) versus fuzzy rules.

* Frequent itemsets of the credit data (training clients of Lab 3) and the effect
  of the Apriori property (candidates versus frequent itemsets per level);
* rules for default with support, confidence and lift; Fisher's exact test of each
  rule with the Benjamini-Hochberg correction;
* a classifier from the rules (the most confident matching rule) compared with the
  Wang-Mendel fuzzy rules on the same four attributes;
* rules for bad risk in the German credit data.

Run:  python -m lab5_fuzzy.run_ext
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from statsmodels.stats.multitest import multipletests

from dmfin import config as C
from dmfin import plotting as P
from dmfin.data import load_credit
from dmfin.data_ext import load_german_credit
from dmfin.reporting import save_numbers, save_table
from lab3_classification.model import prepare

from .apriori import apriori, one_hot, rules
from .fuzzy import Partition, WangMendel

MIN_SUP = 0.02


def terciles(x: pd.Series, ref: pd.Series) -> pd.Series:
    a, b = ref.quantile([1 / 3, 2 / 3])
    return pd.cut(x, [-np.inf, a, b, np.inf], labels=["low", "medium", "high"]).astype(str)


def items_taiwan(X: pd.DataFrame, ref: pd.DataFrame, full: bool) -> pd.DataFrame:
    it = pd.DataFrame(index=X.index)
    it["pay_1"] = np.select([X["pay_1"] <= -1, X["pay_1"] == 0], ["paid", "revolving"], "late")
    for c in ("utilisation", "log_limit", "pay_ratio"):
        it[c.replace("log_", "")] = terciles(X[c], ref[c])
    if full:
        it["delinquent_months"] = pd.cut(X["n_delinquent"], [-1, 0, 2, 6], labels=["0", "1-2", "3+"]).astype(str)
        it["age"] = pd.cut(X["age"], [0, 29, 45, 100], labels=["<30", "30-45", ">45"]).astype(str)
        it["sex"] = np.where(X["female"] == 1, "female", "male")
        it["married"] = np.where(X["married"] == 1, "yes", "no")
    return it


def rule_scores(B: pd.DataFrame, rl: pd.DataFrame, prior: float) -> np.ndarray:
    """Score of each case: confidence of the most confident matching rule (prior if none matches)."""
    score = np.full(len(B), prior)
    for ante, conf in zip(rl["antecedent"], rl["confidence"]):
        cols = ante.split(" AND ")
        if all(c in B.columns for c in cols):
            m = B[cols].all(axis=1).to_numpy()
            score = np.where(m, np.maximum(score, conf), score)
    return score


def main():
    num = {"min_sup": 100 * MIN_SUP}
    X, y = prepare(load_credit())
    Xtrva, Xte, ytrva, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=C.SEED)
    Xtr, Xva, ytr, yva = train_test_split(Xtrva, ytrva, test_size=0.25, stratify=ytrva, random_state=C.SEED)
    prior = float(ytr.mean())

    # ------------------------------------------------ frequent itemsets and rules (all attributes)
    it = items_taiwan(Xtr, Xtr, full=True)
    it["default"] = np.where(ytr == 1, "yes", "no")
    B = one_hot(it)
    freq, lv = apriori(B, MIN_SUP, max_len=4)
    save_table("lab5_apriori_levels", lv.rename(columns={"size": "Itemset size", "candidates": "Candidates",
                                                         "frequent": "Frequent"}).set_index("Itemset size"), nd=0)
    rl = rules(freq, ["default=yes", "default=no"])
    yes = rl[rl["consequent"] == "default=yes"].copy()
    n = len(B)
    pv = []
    for ante in yes["antecedent"]:
        mx = B[ante.split(" AND ")].all(axis=1).to_numpy(); dy = (ytr == 1).to_numpy()
        pv.append(fisher_exact([[np.sum(mx & dy), np.sum(mx & ~dy)], [np.sum(~mx & dy), np.sum(~mx & ~dy)]],
                               alternative="greater")[1])
    yes["p"] = pv
    yes["bh"] = multipletests(yes["p"], 0.05, "fdr_bh")[0]
    top = yes[yes["bh"]].sort_values(["lift", "support"], ascending=False).head(10)
    tab = top[["antecedent", "support", "confidence", "lift"]].copy()
    tab["antecedent"] = tab["antecedent"].str.replace("_", " ").str.replace(" AND ", " $\\wedge$ ")
    tab["support"] *= 100
    tab = tab.rename(columns={"antecedent": "IF (antecedent)", "support": "Support, \\%", "confidence": "Confidence",
                              "lift": "Lift"}).set_index("IF (antecedent)")
    save_table("lab5_assoc_rules", tab, nd={"Support, \\%": 1, "Confidence": 3, "Lift": 2}, col_format="p{9.4cm}ccc")
    num.update({"n_items": B.shape[1], "n_freq": len(freq), "n_rules_yes": len(yes), "n_rules_bh": int(yes["bh"].sum()),
                "n_rules_raw": int((yes["p"] < 0.05).sum()), "top_lift": float(top["lift"].iloc[0]),
                "top_conf": float(top["confidence"].iloc[0]), "top_sup": 100 * float(top["support"].iloc[0]),
                "prior": 100 * prior, "lv2_cand": int(lv.iloc[1]["candidates"]), "lv3_cand": int(lv.iloc[2]["candidates"]),
                "lv3_freq": int(lv.iloc[2]["frequent"]),
                "rules_no_maxconf": float(rl[rl["consequent"] == "default=no"]["confidence"].max())})
    f, ax = P.fig(2.9)
    sc = ax.scatter(100 * yes["support"], yes["confidence"], c=yes["lift"], cmap=P.SEQ, s=8 + 4 * yes["length"],
                    edgecolor=P.INK2, lw=0.2)
    ax.axhline(prior, color=P.MUTED, ls=":", lw=0.9, label="default rate (lift = 1)")
    ax.set_xscale("log"); ax.set_xlabel("support, %"); ax.set_ylabel("confidence"); ax.legend(fontsize=7.5)
    f.colorbar(sc, ax=ax, label="lift")
    P.save(f, "lab5_assoc_rules")

    # ------------------------------------------------ rule classifier vs Wang-Mendel (same four attributes)
    cols = ["pay_1", "utilisation", "log_limit", "pay_ratio"]
    Bs = {k: one_hot(items_taiwan(Xk, Xtr, full=False)) for k, Xk in (("tr", Xtr), ("va", Xva), ("te", Xte))}
    for k in Bs:
        Bs[k] = Bs[k].reindex(columns=Bs["tr"].columns, fill_value=False)
    it4 = Bs["tr"].copy(); it4["default=yes"] = (ytr == 1).to_numpy(); it4["default=no"] = (ytr == 0).to_numpy()
    f4, _ = apriori(it4, 0.005, max_len=5)
    r4 = rules(f4, ["default=yes"])
    s_va, s_te = rule_scores(Bs["va"], r4, prior), rule_scores(Bs["te"], r4, prior)
    q = lambda c, p: float(Xtr[c].quantile(p))
    parts = {"pay_1": Partition(-1.0, 0.0, 2.0, names=("paid", "revolving", "late")),
             "utilisation": Partition(q("utilisation", 0.1), q("utilisation", 0.5), q("utilisation", 0.9)),
             "log_limit": Partition(q("log_limit", 0.1), q("log_limit", 0.5), q("log_limit", 0.9)),
             "pay_ratio": Partition(q("pay_ratio", 0.1), q("pay_ratio", 0.5), q("pay_ratio", 0.9))}
    wm = WangMendel(parts).fit(Xtr[cols], ytr)
    num.update({"rc_rules": len(r4), "rc_auc_val": roc_auc_score(yva, s_va), "rc_auc_test": roc_auc_score(yte, s_te),
                "wm_auc_test": roc_auc_score(yte, wm.predict_proba(Xte[cols])[:, 1]),
                "rc_distinct_scores": int(len(np.unique(s_te)))})

    # ------------------------------------------------ German credit: rules for bad risk
    g = load_german_credit()
    yb = g.pop("bad")
    gi = pd.DataFrame(index=g.index)
    for c in g.columns:
        gi[c] = terciles(g[c], g[c]) if pd.api.types.is_numeric_dtype(g[c]) and g[c].nunique() > 4 else g[c].astype(str)
    gi["risk"] = np.where(yb == 1, "bad", "good")
    Bg = one_hot(gi)
    fg, lvg = apriori(Bg, 0.05, max_len=4)
    rg = rules(fg, ["risk=bad"])
    pg = []
    for ante in rg["antecedent"]:
        mx = Bg[ante.split(" AND ")].all(axis=1).to_numpy(); dy = (yb == 1).to_numpy()
        pg.append(fisher_exact([[np.sum(mx & dy), np.sum(mx & ~dy)], [np.sum(~mx & dy), np.sum(~mx & ~dy)]],
                               alternative="greater")[1])
    rg["p"] = pg; rg["bh"] = multipletests(rg["p"], 0.05, "fdr_bh")[0]
    tg = rg[rg["bh"]].sort_values(["lift", "support"], ascending=False).head(8)
    tabg = tg[["antecedent", "support", "confidence", "lift"]].copy()
    tabg["antecedent"] = tabg["antecedent"].str.replace("_", " ").str.replace(" AND ", " $\\wedge$ ")
    tabg["support"] *= 100
    tabg = tabg.rename(columns={"antecedent": "IF (antecedent)", "support": "Support, \\%", "confidence": "Confidence",
                                "lift": "Lift"}).set_index("IF (antecedent)")
    save_table("lab5_assoc_german", tabg, nd={"Support, \\%": 1, "Confidence": 3, "Lift": 2}, col_format="p{9.4cm}ccc")
    num.update({"de_items": Bg.shape[1], "de_rules": len(rg), "de_rules_bh": int(rg["bh"].sum()),
                "de_rules_raw": int((rg["p"] < 0.05).sum()), "de_top_lift": float(tg["lift"].iloc[0]) if len(tg) else np.nan})
    save_numbers("lab5x", num)
    print(num)


if __name__ == "__main__":
    main()
