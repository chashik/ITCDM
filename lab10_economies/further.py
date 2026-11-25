"""Stage 6: the further methods of chapter 4 (Part 3) applied to the questions of the earlier stages.

  k-medoids     PAMk on the 232 country-months of stage 4 (inflation, its change over 3 months, real policy rate;
                standardized), PAM against k-means, robustness to the most outlying months, CLARA against PAM on the
                612 country-months 2013-2025 of the long panel
  spectral      the groups of series of stage 2 by spectral clustering of |rho| against average linkage
  hmm           Gaussian hidden Markov model of inflation 2013-2025 (four countries, shared parameters)
  fim           monthly events of the country-months: Apriori (Lab 5) against FP-growth, rules; speed on Lab 5's data
  trees         the policy decision of the next month (raise / hold / cut): conditional inference tree against CART and
                "hold" (leave one country out); root selection when the decisions are permuted (no association)
  covering      rules for a rise of the policy rate by sequential covering
  bn            Bayesian network of the past and the next three-month changes
Outputs: results/lab10/q7_*.csv; figures lab10_kmedoids, lab10_hmm, lab10_bn.
Run (from code/):  python -m lab10_economies.further
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier

from further_methods import bayesnet, covering, ctree, fpgrowth, hmm, kmedoids, spectral
from lab5_fuzzy.apriori import apriori, rules

from .common import COUNTRIES, P, panel, save_figure, save_table
from .paths import RESULTS

SEED = 0
REG_FEATS = ["inflation", "change over 3 months", "real policy rate"]
BAD = ("PL", "2023-02")                 # the country-month whose inflation is falsified in the robustness check


# --------------------------------------------------------------------------------------------- data
def months(panels: dict) -> pd.DataFrame:
    """Country-months 2021-03..2025-12 with the quantities of the stage (3-month changes need December 2020 .. February 2021)."""
    rows = []
    for c, p in panels.items():
        food = pd.concat([100 * (p[f"price_{g}"] / p[f"price_{g}"].shift(3) - 1) for g in ("flour_wheat", "milk", "sugar")],
                         axis=1).mean(axis=1)
        pol = p["policy_rate"]
        d = pd.DataFrame({"country": c, "month": p.index.astype(str), "inflation": p["cpi_yoy"],
                          "change over 3 months": p["cpi_yoy"] - p["cpi_yoy"].shift(3),
                          "real policy rate": p["real_policy_rate"], "USD rate y/y": p["fx_usd_yoy"],
                          "USD rate, 3 months": 100 * (p["fx_usd"] / p["fx_usd"].shift(3) - 1),
                          "food, 3 months": food, "petrol, 3 months": 100 * (p["price_petrol"] / p["price_petrol"].shift(3) - 1),
                          "policy change, 3 months": pol - pol.shift(3), "policy change, next month": pol.shift(-1) - pol,
                          "inflation change, next 3 months": p["cpi_yoy"].shift(-3) - p["cpi_yoy"],
                          "policy change, next 3 months": pol.shift(-3) - pol})
        rows.append(d.iloc[3:])                                  # from 2021-03
    return pd.concat(rows, ignore_index=True)


# --------------------------------------------------------------------------------------------- k-medoids
def representatives(X: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """PAM medoids and k-means centres (standardized units), both ordered by inflation."""
    med, _, _, _ = kmedoids.pam(cdist(X, X), k)
    km = KMeans(k, n_init=20, random_state=SEED).fit(X)
    return X[med[np.argsort(X[med, 0])]], km.cluster_centers_[np.argsort(km.cluster_centers_[:, 0])]


def kmedoids_stage(M: pd.DataFrame) -> dict:
    D = M.dropna(subset=REG_FEATS).reset_index(drop=True)
    Xr = D[REG_FEATS].to_numpy(float)
    mu, sd = Xr.mean(0), Xr.std(0)
    X = (Xr - mu) / sd
    Dm = cdist(X, X)
    sil = []
    for k in range(2, 7):
        _, lab, _, _ = kmedoids.pam(Dm, k)
        km = KMeans(k, n_init=20, random_state=SEED).fit_predict(X)
        sil.append({"clusters": k, "PAM": silhouette_score(Dm, lab, metric="precomputed"), "k-means": silhouette_score(X, km)})
    sil = pd.DataFrame(sil).set_index("clusters")
    k_best = int(sil["PAM"].idxmax())

    def fit(Xs, idx):
        med, lab, _, _ = kmedoids.pam(cdist(Xs, Xs), 3)
        o = np.argsort(Xs[med, 0]); med = med[o]
        km = KMeans(3, n_init=20, random_state=SEED).fit(Xs)
        oc = np.argsort(km.cluster_centers_[:, 0])
        return idx[med], km.cluster_centers_[oc], np.argsort(o)[lab], np.argsort(oc)[km.labels_]

    idx_all = np.arange(len(X))
    med, cen, lab_pam, lab_km = fit(X, idx_all)
    rows = []
    for r in range(3):
        m = med[r]
        rows.append({"regime": r + 1, "PAM medoid": f"{D.country[m]} {D.month[m]}", "PAM size": int((lab_pam == r).sum()),
                     **{f"medoid: {f}": Xr[m, j] for j, f in enumerate(REG_FEATS)},
                     "k-means size": int((lab_km == r).sum()),
                     **{f"centre: {f}": cen[r, j] * sd[j] + mu[j] for j, f in enumerate(REG_FEATS)}})
    medoids = pd.DataFrame(rows).set_index("regime")
    # robustness to one wrong value: the inflation of one country-month multiplied by 2, 5 or 10 (an error like the parser
    # error of the data set), all other values and the scaling unchanged; shift of the representatives of the other objects
    bad = int(np.flatnonzero((D.country == BAD[0]) & (D.month == BAD[1]))[0])
    rob = []
    for factor in (2, 5, 10):
        row = {"inflation of " + " ".join(BAD) + " multiplied by": factor}
        for k in (2, 3):
            Xc = Xr.copy(); Xc[bad, 0] *= factor
            X1 = (Xc - mu) / sd
            m0, c0 = representatives(X, k)
            m1, c1 = representatives(X1, k)
            row[f"PAM, k = {k}"] = float(np.linalg.norm(m0 - m1, axis=1).mean())
            row[f"k-means, k = {k}"] = float(np.linalg.norm(c0 - c1, axis=1).mean())
        rob.append(row)
    rob = pd.DataFrame(rob).set_index("inflation of " + " ".join(BAD) + " multiplied by")
    # where the k-means centres lie: distance to the nearest country-month (a medoid is one, distance 0)
    medoids["k-means centre: nearest country-month"] = [f"{D.country[j]} {D.month[j]}" for j in cdist(cen, X).argmin(axis=1)]
    medoids["k-means centre: its distance (s.d.)"] = cdist(cen, X).min(axis=1)
    D["PAM regime"], D["k-means regime"] = lab_pam + 1, lab_km + 1
    return {"D": D, "sil": sil, "k_best": k_best, "medoids": medoids, "robust": rob, "med": med,
            "cen": cen * sd + mu}


def clara_stage() -> pd.DataFrame:
    from .forecasting import long_panel
    rows = []
    for c in COUNTRIES:
        d = long_panel(c)
        rows.append(pd.DataFrame({"inflation": d["y"], "change over 3 months": d["y"] - d["y"].shift(3),
                                  "real policy rate": d["real_policy"]}).dropna())
    Xr = pd.concat(rows).to_numpy(float)
    X = (Xr - Xr.mean(0)) / Xr.std(0)
    t0 = time.perf_counter(); med, _, c_pam, swaps = kmedoids.pam(cdist(X, X), 3); t_pam = time.perf_counter() - t0
    out = [{"method": "PAM, all objects", "objects used by PAM": len(X), "cost relative to PAM": 1.0, "worst of 10 runs": 1.0,
            "medoids shared with PAM": 3, "seconds": t_pam}]
    for s, n_s in ((46, 5), (12, 5), (12, 1)):
        ratios, shared, secs = [], [], []
        for seed in range(10):
            t0 = time.perf_counter()
            m, _, c, _ = kmedoids.clara(X, 3, n_samples=n_s, sample_size=s, seed=seed)
            secs.append(time.perf_counter() - t0)
            ratios.append(c / c_pam); shared.append(len(set(m) & set(med)))
        out.append({"method": f"CLARA, {n_s} sample{'s' if n_s > 1 else ''} of {s}", "objects used by PAM": s,
                    "cost relative to PAM": float(np.mean(ratios)), "worst of 10 runs": float(np.max(ratios)),
                    "medoids shared with PAM": float(np.mean(shared)), "seconds": float(np.mean(secs))})
    t = pd.DataFrame(out).set_index("method")
    t.attrs["n"] = len(X)
    return t


# --------------------------------------------------------------------------------------------- spectral
def spectral_stage(panels: dict) -> pd.DataFrame:
    from .relations import features, groups
    rows = []
    for c in COUNTRIES:
        F = features(panel(c), c)
        corr = F.corr(method="spearman")
        hier = groups(corr)
        lab_h = np.zeros(len(corr), int)
        for g, members in enumerate(hier):
            lab_h[[corr.index.get_loc(m) for m in members]] = g
        W = corr.abs().fillna(0).to_numpy(copy=True)
        np.fill_diagonal(W, 0)
        k_eig = spectral.eigengap(W, kmax=min(5, len(W) - 2))
        lab_s, vals = spectral.njw(W, len(hier), seed=SEED)
        lab_e, _ = spectral.njw(W, k_eig, seed=SEED)
        text = lambda lab: "; ".join(", ".join(corr.index[lab == g]) for g in sorted(set(lab), key=lambda g: -np.sum(lab == g)))
        rows.append({"country": c, "series": len(W), "groups, average linkage": len(hier), "groups by eigengap": k_eig,
                     "adjusted Rand, same number of groups": adjusted_rand_score(lab_h, lab_s),
                     "spectral groups (eigengap)": text(lab_e)})
    return pd.DataFrame(rows).set_index("country")


# --------------------------------------------------------------------------------------------- hidden Markov model
def hmm_stage() -> dict:
    from .forecasting import long_panel
    seqs = {c: long_panel(c)["y"].dropna() for c in COUNTRIES}
    X = [s.to_numpy() for s in seqs.values()]
    n_obs = sum(len(x) for x in X)
    models, bic = {}, []
    for N in (2, 3, 4, 5):
        m = hmm.GaussianHMM(N, n_starts=20, seed=SEED).fit(X)
        models[N] = m
        bic.append({"states": N, "log-likelihood": m.loglik, "parameters": m.n_params, "BIC": m.bic(n_obs)})
    bic = pd.DataFrame(bic).set_index("states")
    m = models[3]
    names = ["low", "moderate", "high"]
    states = pd.DataFrame({"mean inflation, %": m.mu[:, 0], "standard deviation, p.p.": np.sqrt(m.var[:, 0]),
                           "stay probability": np.diag(m.A), "expected stay, months": m.durations()}, index=names)
    trans = pd.DataFrame(m.A, index=[f"from {n}" for n in names], columns=[f"to {n}" for n in names])
    paths = {}
    for c, s in seqs.items():
        q = m.decode(s.to_numpy())
        paths[c] = pd.Series(np.array(names)[q], index=s.index)
    share = pd.DataFrame({c: p.value_counts(normalize=True).reindex(names, fill_value=0) * 100 for c, p in paths.items()}).T
    # comparison with the exact segmentation of stage 3 (2021-2025)
    reg = pd.read_csv(RESULTS / "q3_regimes.csv")
    cmp_rows = []
    for c, p in paths.items():
        q = p.loc["2021-01":"2025-12"]
        switches = int((q != q.shift()).iloc[1:].sum())
        high = q[q == "high"]
        cmp_rows.append({"country": c, "segments of stage 3": int((reg.country == c).sum()), "HMM state changes 2021-2025": switches,
                         "first month in the high state": str(high.index[0]) if len(high) else "--",
                         "last month in the high state": str(high.index[-1]) if len(high) else "--"})
    return {"bic": bic, "states": states, "trans": trans, "paths": paths, "share": share,
            "cmp": pd.DataFrame(cmp_rows).set_index("country"), "seqs": seqs}


# --------------------------------------------------------------------------------------------- frequent itemsets
EVENTS = {
    "inflation above 10 %": lambda d: d["inflation"] > 10,
    "inflation rising": lambda d: d["change over 3 months"] > 1,
    "inflation falling": lambda d: d["change over 3 months"] < -1,
    "policy rate raised": lambda d: d["policy change, 3 months"] > 0.1,
    "policy rate cut": lambda d: d["policy change, 3 months"] < -0.1,
    "real rate negative": lambda d: d["real policy rate"] < 0,
    "dollar up over 10 % y/y": lambda d: d["USD rate y/y"] > 10,
    "food up over 3 %": lambda d: d["food, 3 months"] > 3,
    "petrol up over 3 %": lambda d: d["petrol, 3 months"] > 3,
}
MIN_SUP = 0.10


def fim_stage(M: pd.DataFrame) -> dict:
    B = pd.DataFrame({k: f(M).fillna(False).astype(bool) for k, f in EVENTS.items()})
    t0 = time.perf_counter(); fa, lv = apriori(B, MIN_SUP, max_len=4); ta = time.perf_counter() - t0
    t0 = time.perf_counter(); ff, st = fpgrowth.fpgrowth(B, MIN_SUP, max_len=4); tf = time.perf_counter() - t0
    same = fa.keys() == ff.keys() and all(abs(fa[s] - ff[s]) < 1e-12 for s in fa)
    rl = rules(ff, ["inflation rising", "inflation falling", "policy rate raised", "policy rate cut"], min_conf=0.6)
    rl = rl[(rl["lift"] > 1.2) & (rl["length"] <= 3)].sort_values(["lift", "support"], ascending=False).head(6)
    # the same comparison on the 18 000 transactions of Lab 5 (credit data, minimum support 2 %, itemsets of up to 4 items)
    from dmfin import config as C
    from dmfin.data import load_credit
    from lab3_classification.model import prepare
    from lab5_fuzzy.apriori import one_hot
    from lab5_fuzzy.run_ext import MIN_SUP as SUP5, items_taiwan
    from sklearn.model_selection import train_test_split
    Xc, yc = prepare(load_credit())
    Xtrva, _, ytrva, _ = train_test_split(Xc, yc, test_size=0.2, stratify=yc, random_state=C.SEED)
    Xtr, _, ytr, _ = train_test_split(Xtrva, ytrva, test_size=0.25, stratify=ytrva, random_state=C.SEED)
    it = items_taiwan(Xtr, Xtr, full=True)
    it["default"] = np.where(ytr == 1, "yes", "no")
    B5 = one_hot(it)
    t0 = time.perf_counter(); fa5, lv5 = apriori(B5, SUP5, max_len=4); ta5 = time.perf_counter() - t0
    t0 = time.perf_counter(); ff5, st5 = fpgrowth.fpgrowth(B5, SUP5, max_len=4); tf5 = time.perf_counter() - t0
    same5 = fa5.keys() == ff5.keys() and all(abs(fa5[s] - ff5[s]) < 1e-12 for s in fa5)
    t = pd.DataFrame({
        "Lab 10 events": {"transactions": len(B), "items": B.shape[1], "frequent itemsets": len(ff), "identical results": same,
                          "Apriori: database passes": len(lv), "Apriori: candidates counted": int(lv["candidates"].sum()),
                          "FP-growth: database passes": 2, "FP-growth: nodes of the FP-tree": st["tree nodes"],
                          "FP-growth: conditional trees": st["conditional trees"], "Apriori, seconds": ta, "FP-growth, seconds": tf},
        "Lab 5 credit data": {"transactions": len(B5), "items": B5.shape[1], "frequent itemsets": len(ff5), "identical results": same5,
                              "Apriori: database passes": len(lv5), "Apriori: candidates counted": int(lv5["candidates"].sum()),
                              "FP-growth: database passes": 2, "FP-growth: nodes of the FP-tree": st5["tree nodes"],
                              "FP-growth: conditional trees": st5["conditional trees"], "Apriori, seconds": ta5, "FP-growth, seconds": tf5}})
    support = B.mean().mul(100).rename("share of country-months, %")
    return {"B": B, "fim": t, "rules": rl, "support": support}


# --------------------------------------------------------------------------------------------- trees and rules
TREE_FEATS = ["inflation", "change over 3 months", "real policy rate", "USD rate y/y", "food, 3 months"]


def decisions(M: pd.DataFrame) -> pd.DataFrame:
    d = M.dropna(subset=TREE_FEATS + ["policy change, next month"]).reset_index(drop=True)
    ch = d["policy change, next month"]
    d["decision"] = np.where(ch > 0.05, "raise", np.where(ch < -0.05, "cut", "hold"))
    return d


def cart(X, y) -> DecisionTreeClassifier:
    """CART pruned by cost complexity, alpha chosen by 5-fold cross-validated accuracy (largest alpha among the best)."""
    path = DecisionTreeClassifier(min_samples_leaf=7, random_state=SEED).cost_complexity_pruning_path(X, y)
    alphas = np.unique(path.ccp_alphas)
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    acc = [cross_val_score(DecisionTreeClassifier(min_samples_leaf=7, ccp_alpha=a, random_state=SEED), X, y, cv=cv).mean()
           for a in alphas]
    a = alphas[np.flatnonzero(np.isclose(acc, max(acc)))[-1]]
    return DecisionTreeClassifier(min_samples_leaf=7, ccp_alpha=a, random_state=SEED).fit(X, y)


def trees_stage(M: pd.DataFrame) -> dict:
    d = decisions(M)
    X, y = d[TREE_FEATS].to_numpy(float), d["decision"].to_numpy()
    ct = ctree.CTree(alpha=0.05).fit(X, y, names=TREE_FEATS, classify=True)
    cr = cart(X, y)
    loco = []
    for c in COUNTRIES:
        tr, te = (d.country != c).to_numpy(), (d.country == c).to_numpy()
        maj = pd.Series(y[tr]).mode()[0]
        t1 = ctree.CTree(alpha=0.05).fit(X[tr], y[tr], names=TREE_FEATS, classify=True)
        t2 = cart(X[tr], y[tr])
        loco.append({"country left out": c, "months": int(te.sum()), "\"hold\" (majority)": float((y[te] == maj).mean()),
                     "conditional inference tree": float((t1.predict(X[te]) == y[te]).mean()),
                     "CART": float((t2.predict(X[te]) == y[te]).mean())})
    loco = pd.DataFrame(loco).set_index("country left out")
    loco.loc["all four"] = [int(loco["months"].sum())] + [float((loco[col] * loco["months"]).sum() / loco["months"].sum())
                                                          for col in loco.columns[1:]]
    from sklearn.model_selection import RepeatedStratifiedKFold
    acc = {"\"hold\" (majority)": [], "conditional inference tree": [], "CART": []}
    for tr, te in RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=SEED).split(X, y):
        acc["\"hold\" (majority)"].append((y[te] == pd.Series(y[tr]).mode()[0]).mean())
        acc["conditional inference tree"].append((ctree.CTree(alpha=0.05).fit(X[tr], y[tr], names=TREE_FEATS, classify=True).predict(X[te]) == y[te]).mean())
        acc["CART"].append((cart(X[tr], y[tr]).predict(X[te]) == y[te]).mean())
    loco.loc["5-fold cross-validation, 5 repeats"] = [len(y)] + [float(np.mean(v)) for v in acc.values()]
    ct_rules = pd.DataFrame([{"leaf": r, "months": n, **{f"P({k})": v for k, v in zip(ct.classes, p)}}
                             for r, n, p in ct.rules(lambda v: f"{v:.1f}")])
    # selection of the root variable when the decisions carry no information (permuted): 300 permutations
    rng = np.random.default_rng(SEED)
    names = TREE_FEATS + ["noise, continuous", "noise, binary"]
    cnt = {"CART": dict.fromkeys(names, 0), "conditional inference tree": dict.fromkeys(names, 0)}
    splits = 0
    n_perm = 300
    for _ in range(n_perm):
        Xa = np.c_[X, rng.normal(size=len(X)), rng.integers(0, 2, len(X))]
        yp = rng.permutation(y)
        stump = DecisionTreeClassifier(max_depth=1, random_state=SEED).fit(Xa, yp)
        cnt["CART"][names[stump.tree_.feature[0]]] += 1
        t = ctree.CTree(alpha=0.05).fit(Xa, yp, names=names, classify=True)
        best = min(t.root.tests, key=lambda r: r[1])[0] if t.root.tests else None
        if best:
            cnt["conditional inference tree"][best] += 1
        splits += t.root.var is not None
    bias = pd.DataFrame(cnt).div(n_perm).mul(100)
    bias.loc["root split made, % of permutations"] = [100.0, 100 * splits / n_perm]
    return {"d": d, "ctree": ct, "cart": cr, "loco": loco, "ct_rules": ct_rules, "bias": bias,
            "cart_depth": int(cr.get_depth()), "cart_leaves": int(cr.get_n_leaves()),
            "ct_splits": ct.splits(), "ct_root_p": ct.root.pvalue}


def covering_stage(d: pd.DataFrame) -> pd.DataFrame:
    X, y = d[TREE_FEATS].to_numpy(float), d["decision"].to_numpy()
    rows = []
    for target in ("raise", "cut"):
        sc = covering.SequentialCovering(min_pos=5, min_accuracy=0.7, max_conds=3).fit(X, y, target, names=TREE_FEATS)
        for txt, r in zip(sc.text(lambda v: f"{v:.1f}"), sc.rules_):
            rows.append({"class": target, "rule": txt, "months covered": r["positives"] + r["negatives"],
                         "of them in the class": r["positives"], "accuracy": r["positives"] / (r["positives"] + r["negatives"])})
        cov = sc.covers(X)
        rows.append({"class": target, "rule": "all rules together", "months covered": int(cov.sum()),
                     "of them in the class": int((cov & (y == target)).sum()),
                     "accuracy": float((cov & (y == target)).sum() / max(cov.sum(), 1))})
        rows[-1]["share of the class covered"] = float((cov & (y == target)).sum() / (y == target).sum())
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------------- Bayesian network
def three(x, lo, hi, names):
    return np.where(x < lo, names[0], np.where(x > hi, names[2], names[1]))


def bn_data(M: pd.DataFrame) -> pd.DataFrame:
    d = M.dropna(subset=["USD rate, 3 months", "food, 3 months", "change over 3 months", "policy change, 3 months",
                         "inflation change, next 3 months", "policy change, next 3 months"])
    return pd.DataFrame({
        "USD rate, past": three(d["USD rate, 3 months"], -3, 3, ["down", "flat", "up"]),
        "food, past": three(d["food, 3 months"], 0, 3, ["down", "moderate", "up"]),
        "inflation, past": three(d["change over 3 months"], -1, 1, ["falling", "stable", "rising"]),
        "policy, past": three(d["policy change, 3 months"], -0.1, 0.1, ["cut", "hold", "raise"]),
        "inflation, next": three(d["inflation change, next 3 months"], -1, 1, ["falling", "stable", "rising"]),
        "policy, next": three(d["policy change, next 3 months"], -0.1, 0.1, ["cut", "hold", "raise"])})


def bn_stage(M: pd.DataFrame) -> dict:
    B = bn_data(M)
    past = [c for c in B if c.endswith("past")]
    nxt = [c for c in B if c.endswith("next")]
    black = {(a, b) for a in nxt for b in past}                 # expert knowledge: the next months cannot cause the past
    parents, score, moves = bayesnet.hill_climb(B, blacklist=black, max_parents=2)
    parents_aic, _, moves_aic = bayesnet.hill_climb(B, blacklist=black, max_parents=2, penalty="aic")
    empty = sum(bayesnet.node_score(B, v, (), {c: sorted(B[c].unique()) for c in B}) for v in B)
    arcs_b = {(p, v) for v, ps in parents.items() for p in ps}
    arcs_a = {(p, v) for v, ps in parents_aic.items() for p in ps}
    arcs = pd.DataFrame([{"arc": f"{p} → {v}", "BIC": "yes" if (p, v) in arcs_b else "no", "AIC": "yes" if (p, v) in arcs_a else "no"}
                         for p, v in sorted(arcs_b | arcs_a, key=lambda e: (e not in arcs_b, e))])
    cpt = bayesnet.cpts(B, parents, alpha=1.0)
    J = bayesnet.joint(cpt, parents)
    J_aic = bayesnet.joint(bayesnet.cpts(B, parents_aic, alpha=1.0), parents_aic)
    q = []
    for target, value, ev in [("inflation, next", "rising", {}), ("inflation, next", "rising", {"inflation, past": "rising"}),
                              ("inflation, next", "rising", {"USD rate, past": "up"}),
                              ("inflation, next", "rising", {"USD rate, past": "up", "food, past": "up"}),
                              ("policy, next", "raise", {}), ("policy, next", "raise", {"inflation, past": "rising"}),
                              ("policy, next", "cut", {"inflation, past": "falling"})]:
        emp = B
        for k, v in ev.items():
            emp = emp[emp[k] == v]
        q.append({"query": f"P({target} = {value}" + (" | " + ", ".join(f"{k} = {v}" for k, v in ev.items()) if ev else "") + ")",
                  "network (BIC)": float(bayesnet.query(J, target, ev).get(value, 0.0)),
                  "network (AIC)": float(bayesnet.query(J_aic, target, ev).get(value, 0.0)),
                  "share in the data": float((emp[target] == value).mean()) if len(emp) else np.nan, "months": len(emp)})
    return {"B": B, "parents": parents, "parents_aic": parents_aic, "arcs": arcs, "score": score, "empty": empty, "queries": pd.DataFrame(q)}


# --------------------------------------------------------------------------------------------- figures
def figures(km: dict, hm: dict, bn: dict):
    D = km["D"]
    f, ax = P.fig(3.3)
    cols = [P.BLUE, P.ORANGE, P.RED]
    for r in range(3):
        s = D[D["PAM regime"] == r + 1]
        ax.scatter(s["inflation"], s["real policy rate"], s=10, color=cols[r], alpha=0.6, label=f"PAM regime {r + 1}")
    for r, m in enumerate(km["med"]):
        ax.scatter(D["inflation"][m], D["real policy rate"][m], s=90, marker="D", color=cols[r], edgecolor="black", zorder=5)
        ax.annotate(f"{D.country[m]} {D.month[m]}", (D["inflation"][m], D["real policy rate"][m]), xytext=(10, 14),
                    textcoords="offset points", fontsize=9, arrowprops=dict(arrowstyle="-", color=P.INK, lw=0.5),
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85), zorder=7)
    for r in range(3):
        ax.scatter(km["cen"][r, 0], km["cen"][r, 2], s=90, marker="X", color="black", zorder=6,
                   label="k-means centre" if r == 0 else None)
    ax.axhline(0, color=P.INK, lw=0.6)
    ax.set_xlabel("inflation, %"); ax.set_ylabel("real policy rate, p.p.")
    ax.legend(fontsize=9, loc="lower left")
    f.tight_layout(); save_figure(f, "q7_kmedoids")

    f, axs = P.fig(4.6, nrows=2, ncols=2, sharex=True)
    shade = {"low": "#ffffff", "moderate": "#fde7c8", "high": "#f6b4a8"}
    for ax, (c, s) in zip(axs.ravel(), hm["seqs"].items()):
        p = hm["paths"][c]
        x = s.index.to_timestamp()
        for i, st in enumerate(p.to_numpy()):
            ax.axvspan(x[i], x[i] + pd.offsets.MonthEnd(1), color=shade[st], lw=0)
        ax.plot(x, s.to_numpy(), color=P.INK, lw=1.0)
        ax.set_title(COUNTRIES[c], fontsize=9.5); ax.set_ylabel("inflation, %", fontsize=9)
    from matplotlib.patches import Patch
    f.legend(handles=[Patch(facecolor=shade[k], edgecolor=P.INK, lw=0.4, label=f"{k} state") for k in shade],
             fontsize=9, loc="upper center", ncol=3, frameon=False)
    f.tight_layout(rect=(0, 0, 1, 0.94)); save_figure(f, "q7_hmm")

    pos = {"USD rate, past": (0, 3), "food, past": (0, 2), "inflation, past": (0, 1), "policy, past": (0, 0),
           "inflation, next": (2.2, 1.9), "policy, next": (2.2, 0.6)}
    f, ax = P.fig(3.2, w=P.TEXT_W * 0.8)
    boxes = {}
    for v, (x, y) in pos.items():
        boxes[v] = ax.text(x, y, v, ha="center", va="center", fontsize=9, zorder=3,
                           bbox=dict(boxstyle="round,pad=0.45", fc="#eaf1fb" if v.endswith("past") else "#fde7c8", ec=P.INK, lw=0.7))
    for v, ps in bn["parents"].items():
        for q in ps:                                      # arrows between the edges of the boxes
            (xq, yq), (xv, yv) = pos[q], pos[v]
            if xq != xv:
                src, dst, cs = (1, 0.5), (0, 0.5), "arc3,rad=0"
            elif abs(yq - yv) <= 1:
                src, dst, cs = ((0.5, 0), (0.5, 1), "arc3,rad=0") if yq > yv else ((0.5, 1), (0.5, 0), "arc3,rad=0")
            else:                                         # skips a box of the same column: around its left side
                src, dst, cs = (0, 0.5), (0, 0.5), "arc3,rad=0.45" if yq > yv else "arc3,rad=-0.45"
            ax.annotate("", xy=dst, xycoords=boxes[v], xytext=src, textcoords=boxes[q],
                        arrowprops=dict(arrowstyle="-|>", mutation_scale=14, lw=1.3, color=P.INK, shrinkA=2, shrinkB=2,
                                        connectionstyle=cs))
    ax.set_xlim(-0.8, 3.0); ax.set_ylim(-0.6, 3.6); ax.axis("off")
    ax.text(0, 3.55, "past 3 months", ha="center", fontsize=9, style="italic")
    ax.text(2.2, 3.55, "next 3 months", ha="center", fontsize=9, style="italic")
    f.tight_layout(); save_figure(f, "q7_bn")


# --------------------------------------------------------------------------------------------- main
def main():
    panels = {c: panel(c, base=True) for c in COUNTRIES}
    M = months(panels)
    km = kmedoids_stage(M)
    cl = clara_stage()
    sp = spectral_stage(panels)
    hm = hmm_stage()
    fi = fim_stage(M)
    tr = trees_stage(M)
    cv = covering_stage(tr["d"])
    bn = bn_stage(M)
    save_table(km["sil"], "q7_kmedoids", 3); save_table(km["medoids"], "q7_medoids"); save_table(km["robust"], "q7_robust", 3)
    save_table(cl, "q7_clara", 4); save_table(sp, "q7_spectral", 3)
    save_table(hm["bic"], "q7_hmm_bic"); save_table(hm["states"], "q7_hmm_states", 3); save_table(hm["trans"], "q7_hmm_transitions", 3)
    save_table(hm["share"], "q7_hmm_share"); save_table(hm["cmp"], "q7_hmm_compare")
    save_table(pd.DataFrame(hm["paths"]), "q7_hmm_paths")
    save_table(fi["fim"], "q7_fim", 3); save_table(fi["rules"], "q7_rules", 3); save_table(fi["support"].to_frame(), "q7_events", 1)
    save_table(tr["loco"], "q7_trees", 3); save_table(tr["ct_rules"], "q7_ctree_rules", 3); save_table(tr["bias"], "q7_bias", 1)
    save_table(cv, "q7_covering", 3)
    save_table(bn["arcs"], "q7_bn_arcs", 2); save_table(bn["queries"], "q7_bn_queries", 3)
    extra = {"pamk_k": km["k_best"], "bad_month": " ".join(BAD), "clara_n": cl.attrs["n"],
             "decisions": tr["d"]["decision"].value_counts().to_dict(), "ct_splits": tr["ct_splits"], "ct_root_p": tr["ct_root_p"],
             "cart_depth": tr["cart_depth"], "cart_leaves": tr["cart_leaves"], "bn_score": bn["score"], "bn_empty": bn["empty"],
             "bn_months": len(bn["B"]), "regime_months": len(km["D"])}
    (RESULTS / "q7_extra.json").write_text(json.dumps(extra, indent=1, default=str))
    figures(km, hm, bn)
    pd.set_option("display.width", 220)
    for name, t in [("silhouette", km["sil"]), ("medoids", km["medoids"]), ("robustness", km["robust"]), ("CLARA", cl),
                    ("spectral", sp), ("HMM BIC", hm["bic"]), ("HMM states", hm["states"]), ("HMM transitions", hm["trans"]),
                    ("HMM share", hm["share"]), ("HMM vs segmentation", hm["cmp"]), ("FIM", fi["fim"]), ("rules", fi["rules"]),
                    ("events", fi["support"]), ("trees LOCO", tr["loco"]), ("ctree rules", tr["ct_rules"]), ("bias", tr["bias"]),
                    ("covering", cv), ("BN arcs", bn["arcs"]), ("BN queries", bn["queries"])]:
        print(f"== {name}\n{t.round(3).to_string() if hasattr(t, 'round') else t}\n")
    print(json.dumps(extra, indent=1, default=str))


if __name__ == "__main__":
    main()
