"""Lab 4. Decision trees and tree ensembles.

A. Classification of credit default (the split of Lab 3): the unpruned tree and
   cost-complexity pruning (cross-validation, one-standard-error rule), impurity
   measures, rules of a shallow tree, random forests and out-of-bag error, gradient
   boosting, three kinds of feature importance, SHAP explanations, a test comparison
   with the models of Lab 3, and the German credit data.
B. Regression trees for next-month volatility (the data of Lab 2): accuracy and the
   behaviour outside the training range.
C. One gradient-boosting model for 62 series (global model) against models fitted
   to each series separately.

Run:  python -m lab4_trees.run
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import (HistGradientBoostingClassifier, HistGradientBoostingRegressor, RandomForestClassifier,
                              RandomForestRegressor)
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, export_text, plot_tree

from dmfin import config as C
from dmfin import plotting as P
from dmfin.data import load_credit, load_market_panel
from dmfin.data_ext import GERMAN_COST, load_german_credit
from dmfin.features import add_targets, market_features
from dmfin.panel import FEATURES as PFEAT, build_panel, split_masks
from dmfin.reporting import save_numbers, save_table
from dmfin.splits import chrono_split
from lab2_regression.model import FEATURES as VFEAT, HORIZON, TARGET, design, make_model
from lab3_classification.model import best_threshold, expected_cost, prepare

C_FN, C_FP = 5.0, 1.0


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def credit_split():
    X, y = prepare(load_credit())
    Xtrva, Xte, ytrva, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=C.SEED)
    Xtr, Xva, ytr, yva = train_test_split(Xtrva, ytrva, test_size=0.25, stratify=ytrva, random_state=C.SEED)
    return Xtr, Xva, Xte, ytr.to_numpy(), yva.to_numpy(), yte.to_numpy()


# =========================================================================== A
def pruning(num, Xtr, Xva, ytr, yva):
    full = DecisionTreeClassifier(random_state=C.SEED).fit(Xtr, ytr)
    num.update({"full_leaves": int(full.get_n_leaves()), "full_depth": int(full.get_depth()),
                "full_auc_train": roc_auc_score(ytr, full.predict_proba(Xtr)[:, 1]),
                "full_auc_val": roc_auc_score(yva, full.predict_proba(Xva)[:, 1])})
    path = full.cost_complexity_pruning_path(Xtr, ytr)
    alphas = np.unique(np.quantile(path.ccp_alphas[path.ccp_alphas > 0], np.linspace(0, 0.995, 40)))
    skf = StratifiedKFold(5, shuffle=True, random_state=C.SEED)
    rows = []
    for a in alphas:
        cv = [roc_auc_score(ytr[v], DecisionTreeClassifier(ccp_alpha=a, random_state=C.SEED).fit(Xtr.iloc[t], ytr[t])
                            .predict_proba(Xtr.iloc[v])[:, 1]) for t, v in skf.split(Xtr, ytr)]
        tree = DecisionTreeClassifier(ccp_alpha=a, random_state=C.SEED).fit(Xtr, ytr)
        rows.append((a, tree.get_n_leaves(), np.mean(cv), np.std(cv) / np.sqrt(5),
                     roc_auc_score(ytr, tree.predict_proba(Xtr)[:, 1]), roc_auc_score(yva, tree.predict_proba(Xva)[:, 1])))
    pr = pd.DataFrame(rows, columns=["alpha", "leaves", "cv", "se", "train", "val"])
    best = pr.loc[pr["cv"].idxmax()]
    ose = pr[pr["cv"] >= best["cv"] - best["se"]].sort_values("leaves").iloc[0]     # one-standard-error rule
    num.update({"cv_best_leaves": int(best["leaves"]), "cv_best_auc": float(best["cv"]), "ose_leaves": int(ose["leaves"]),
                "ose_alpha": float(ose["alpha"]), "ose_auc_cv": float(ose["cv"]), "ose_auc_val": float(ose["val"])})
    f, axs = P.fig(2.8, ncols=2)
    axs[0].plot(pr["alpha"], pr["leaves"], "o-", ms=2.5, color=P.BLUE)
    axs[0].set_xscale("log"); axs[0].set_yscale("log"); axs[0].set_xlabel("complexity parameter $\\alpha$")
    axs[0].set_ylabel("number of leaves")
    axs[1].plot(pr["leaves"], pr["train"], "o-", ms=2.5, color=P.ORANGE, label="training")
    axs[1].errorbar(pr["leaves"], pr["cv"], yerr=pr["se"], fmt="o-", ms=2.5, color=P.BLUE, label="5-fold CV", capsize=1.5)
    axs[1].plot(pr["leaves"], pr["val"], "s--", ms=2.5, color=P.AQUA, label="validation")
    axs[1].axvline(ose["leaves"], color=P.MUTED, ls=":", lw=0.9)
    axs[1].set_xscale("log"); axs[1].set_xlabel("number of leaves"); axs[1].set_ylabel("AUC"); axs[1].legend(fontsize=9)
    f.tight_layout(); P.save(f, "lab4_pruning")
    return float(ose["alpha"])


def impurity_and_rules(num, Xtr, Xva, ytr, yva, alpha):
    rows = []
    for crit in ("gini", "entropy", "log_loss"):
        t = DecisionTreeClassifier(criterion=crit, ccp_alpha=alpha, random_state=C.SEED).fit(Xtr, ytr)
        rows.append((crit, t.get_n_leaves(), roc_auc_score(yva, t.predict_proba(Xva)[:, 1])))
    num.update({f"crit_{r[0]}_auc": r[2] for r in rows})
    # shallow tree (depth 3): its leaves are readable rules
    sh = DecisionTreeClassifier(max_depth=3, random_state=C.SEED).fit(Xtr, ytr)
    f, ax = P.fig(3.6)
    f.subplots_adjust(left=0, right=1, top=1, bottom=0)     # plot_tree spaces the boxes by the width of the axes
    plot_tree(sh, feature_names=list(Xtr.columns), class_names=["pays", "defaults"], filled=False, impurity=False,
              proportion=True, precision=2, fontsize=9, label="root", ax=ax)       # labels at the root only
    P.save(f, "lab4_tree")
    leaf = sh.apply(Xtr)
    rules = export_text(sh, feature_names=list(Xtr.columns), show_weights=False)
    (C.RESULTS_DIR / "lab4_rules.txt").write_text(rules)
    stats = pd.DataFrame({"leaf": leaf, "y": ytr}).groupby("leaf")["y"].agg(["size", "mean"]).sort_values("mean")
    num.update({"shallow_auc_val": roc_auc_score(yva, sh.predict_proba(Xva)[:, 1]), "shallow_leaves": int(sh.get_n_leaves()),
                "leaf_min_rate": 100 * float(stats["mean"].iloc[0]), "leaf_max_rate": 100 * float(stats["mean"].iloc[-1]),
                "leaf_max_size": int(stats["size"].iloc[-1]), "root_feature": Xtr.columns[sh.tree_.feature[0]].replace("_", "\\_"),
                "root_threshold": float(sh.tree_.threshold[0])})


def forest(num, Xtr, Xva, ytr, yva):
    sizes = [10, 25, 50, 100, 200, 300, 500]
    f, ax = P.fig(2.7)
    oob = {}
    for mf, col, name in (("sqrt", P.BLUE, "$m=\\sqrt{p}$"), (0.5, P.ORANGE, "$m=p/2$"), (None, P.AQUA, "$m=p$ (bagging)")):
        rf = RandomForestClassifier(n_estimators=10, max_features=mf, min_samples_leaf=5, oob_score=True, warm_start=True,
                                    n_jobs=-1, random_state=C.SEED)
        errs = []
        for n in sizes:
            rf.set_params(n_estimators=n); rf.fit(Xtr, ytr)
            errs.append(1 - rf.oob_score_)
        oob[name] = errs
        ax.plot(sizes, 100 * np.array(errs), "o-", ms=3, color=col, label=name)
    ax.set_xscale("log"); ax.set_xlabel("number of trees $B$"); ax.set_ylabel("out-of-bag error, %"); ax.legend(fontsize=9)
    P.save(f, "lab4_forest")
    rf = RandomForestClassifier(500, max_features="sqrt", min_samples_leaf=5, oob_score=True, n_jobs=-1,
                                random_state=C.SEED).fit(Xtr, ytr)
    oob_auc = roc_auc_score(ytr, rf.oob_decision_function_[:, 1])
    num.update({"rf_oob_err": 100 * (1 - rf.oob_score_), "rf_oob_auc": oob_auc,
                "rf_val_auc": roc_auc_score(yva, rf.predict_proba(Xva)[:, 1]),
                "rf_oob_err_10": 100 * oob["$m=\\sqrt{p}$"][0], "rf_oob_err_bag": 100 * oob["$m=p$ (bagging)"][-1]})
    return rf


def boosting(num, Xtr, Xva, ytr, yva):
    f, ax = P.fig(2.7)
    curves = {}
    for lr, col in ((0.3, P.RED), (0.1, P.ORANGE), (0.03, P.BLUE)):
        gb = HistGradientBoostingClassifier(learning_rate=lr, max_iter=600, early_stopping=False, random_state=C.SEED).fit(Xtr, ytr)
        loss = [-np.mean(yva * np.log(p[:, 1] + 1e-12) + (1 - yva) * np.log(1 - p[:, 1] + 1e-12))
                for p in gb.staged_predict_proba(Xva)]
        curves[lr] = loss
        ax.plot(np.arange(1, len(loss) + 1), loss, color=col, lw=1.1, label=f"learning rate {lr}")
    ax.set_xscale("log"); ax.set_xlabel("boosting iterations"); ax.set_ylabel("validation log-loss")
    ax.legend(fontsize=9, loc="lower center", bbox_to_anchor=(0.5, 1), ncol=3)
    ax.set_ylim(min(min(c) for c in curves.values()) - 0.005, 0.47)
    P.save(f, "lab4_boosting")
    best = {lr: (int(np.argmin(c)) + 1, float(np.min(c))) for lr, c in curves.items()}
    gb = HistGradientBoostingClassifier(learning_rate=0.1, max_iter=best[0.1][0], early_stopping=False,
                                        random_state=C.SEED).fit(Xtr, ytr)
    num.update({"gb_iter_03": best[0.3][0], "gb_iter_01": best[0.1][0], "gb_iter_003": best[0.03][0],
                "gb_loss_03": best[0.3][1], "gb_loss_01": best[0.1][1], "gb_loss_003": best[0.03][1],
                "gb_loss_03_end": float(curves[0.3][-1]), "gb_val_auc": roc_auc_score(yva, gb.predict_proba(Xva)[:, 1])})
    return gb


def importance(num, rf, gb, Xtr, Xva, ytr, yva):
    rng = np.random.default_rng(C.SEED)
    Xn, Xvn = Xtr.copy(), Xva.copy()
    Xn["noise_id"] = rng.permutation(len(Xn)).astype(float)          # unique values, no information
    Xvn["noise_id"] = rng.permutation(len(Xvn)).astype(float)
    rfn = RandomForestClassifier(300, max_features="sqrt", min_samples_leaf=1, n_jobs=-1, random_state=C.SEED).fit(Xn, ytr)
    imp = pd.Series(rfn.feature_importances_, index=Xn.columns)
    perm = permutation_importance(rfn, Xvn, yva, scoring="roc_auc", n_repeats=5, random_state=C.SEED, n_jobs=-1)
    pimp = pd.Series(perm.importances_mean, index=Xn.columns)
    ex = shap.TreeExplainer(gb)
    sv = ex.shap_values(Xva)
    simp = pd.Series(np.abs(sv).mean(0), index=Xva.columns)
    top = simp.sort_values(ascending=False).index[:10]
    num.update({"noise_imp_rank": int(imp.rank(ascending=False)["noise_id"]), "noise_imp": float(imp["noise_id"]),
                "noise_perm": float(pimp["noise_id"]), "shap_top1": top[0].replace("_", "\\_"),
                "shap_top2": top[1].replace("_", "\\_"), "shap_top3": top[2].replace("_", "\\_")})
    cols = list(imp.sort_values(ascending=False).index[:12])
    if "noise_id" not in cols:
        cols[-1] = "noise_id"
    f, axs = P.fig(3.2, ncols=3, sharey=True)
    for ax, s, title in ((axs[0], imp, "impurity (random forest)"), (axs[1], pimp, "permutation, $\\Delta$AUC"),
                         (axs[2], simp.reindex(cols).fillna(0), "mean $|$SHAP$|$ (boosting)")):
        v = s.reindex(cols).fillna(0)
        ax.barh(range(len(cols)), v.to_numpy(), color=[P.RED if c == "noise_id" else P.BLUE for c in cols])
        ax.set_title(title, fontsize=9)
    axs[0].set_yticks(range(len(cols))); axs[0].set_yticklabels(cols, fontsize=9); axs[0].invert_yaxis()
    f.tight_layout(); P.save(f, "lab4_importance")
    # SHAP summary (beeswarm-type) and the explanation of one client
    f, axs = P.fig(3.6, ncols=2, gridspec_kw={"width_ratios": [1.3, 1]})
    for i, c in enumerate(top[::-1]):
        j = list(Xva.columns).index(c)
        x = Xva[c].to_numpy(); z = (x - np.nanpercentile(x, 5)) / (np.nanpercentile(x, 95) - np.nanpercentile(x, 5) + 1e-12)
        jitter = np.random.default_rng(i).uniform(-0.3, 0.3, len(x))
        axs[0].scatter(sv[:, j], i + jitter, c=np.clip(z, 0, 1), cmap=P.DIV, s=1.5, rasterized=True)
    axs[0].set_yticks(range(len(top))); axs[0].set_yticklabels(list(top[::-1]), fontsize=9)
    axs[0].axvline(0, color=P.INK2, lw=0.6); axs[0].set_xlabel("SHAP value (log-odds); colour: low $\\to$ high feature value")
    k = int(np.argmax(gb.predict_proba(Xva)[:, 1]))
    contrib = pd.Series(sv[k], index=Xva.columns)
    main = contrib.reindex(contrib.abs().sort_values(ascending=False).index[:8])
    rest = contrib.sum() - main.sum()
    vals = list(main.to_numpy()) + [rest]
    labels = [f"{c} = {Xva.iloc[k][c]:.2f}" for c in main.index] + ["other features"]
    axs[1].barh(range(len(vals)), vals, color=[P.RED if v > 0 else P.BLUE for v in vals])
    axs[1].set_yticks(range(len(vals))); axs[1].set_yticklabels(labels, fontsize=9); axs[1].invert_yaxis()
    axs[1].axvline(0, color=P.INK2, lw=0.6); axs[1].set_xlabel("contribution to the log-odds")
    base = float(np.ravel(ex.expected_value)[0])
    axs[1].set_title(f"base {base:.2f} + contributions = {base + contrib.sum():.2f}".replace("-", "−"), fontsize=9)
    f.tight_layout(); P.save(f, "lab4_shap")
    p_k = float(gb.predict_proba(Xva.iloc[[k]])[:, 1][0])
    num.update({"shap_base": base, "shap_client_logit": base + float(contrib.sum()), "shap_client_p": 100 * p_k,
                "shap_additivity": float(abs(base + contrib.sum() - np.log(p_k / (1 - p_k))))})


def compare(num, cart_alpha, rf, gb, Xtr, Xva, Xte, ytr, yva, yte):
    sc = StandardScaler().fit(Xtr)
    lr = LogisticRegression(C=10, max_iter=3000).fit(sc.transform(Xtr), ytr)
    cart = DecisionTreeClassifier(ccp_alpha=cart_alpha, random_state=C.SEED).fit(Xtr, ytr)
    models = {"Logistic regression (Lab 3)": lambda X: lr.predict_proba(sc.transform(X))[:, 1],
              "Decision tree, pruned (CART)": lambda X: cart.predict_proba(X)[:, 1],
              "Random forest (500 trees)": lambda X: rf.predict_proba(X)[:, 1],
              "Gradient boosting": lambda X: gb.predict_proba(X)[:, 1]}
    rows, pt = [], {}
    for name, fpred in models.items():
        pv, pt[name] = fpred(Xva), fpred(Xte)
        thr, _, _ = best_threshold(yva, pv, C_FN, C_FP)
        rows.append((name, roc_auc_score(yte, pt[name]), average_precision_score(yte, pt[name]),
                     brier_score_loss(yte, pt[name]), expected_cost(yte, pt[name], thr, C_FN, C_FP)))
    tab = pd.DataFrame(rows, columns=["Model", "AUC", "PR AUC", "Brier", "Cost*"]).set_index("Model")
    save_table("lab4_compare", tab, nd=3, col_format="p{6.0cm}cccc")
    rng = np.random.default_rng(C.SEED)
    d = []
    for _ in range(1000):
        i = rng.integers(0, len(yte), len(yte))
        if yte[i].min() == yte[i].max():
            continue
        d.append(roc_auc_score(yte[i], pt["Gradient boosting"][i]) - roc_auc_score(yte[i], pt["Logistic regression (Lab 3)"][i]))
    num.update({"cmp_auc_lr": tab.iloc[0]["AUC"], "cmp_auc_cart": tab.iloc[1]["AUC"], "cmp_auc_rf": tab.iloc[2]["AUC"],
                "cmp_auc_gb": tab.iloc[3]["AUC"], "cmp_cost_lr": tab.iloc[0]["Cost*"], "cmp_cost_gb": tab.iloc[3]["Cost*"],
                "cmp_diff_lo": float(np.quantile(d, 0.025)), "cmp_diff_hi": float(np.quantile(d, 0.975))})


def german(num):
    g = load_german_credit()
    y = g.pop("bad").to_numpy()
    cat = [c for c in g.columns if not pd.api.types.is_numeric_dtype(g[c])]
    enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(g[cat])
    X = np.hstack([g[[c for c in g.columns if c not in cat]].to_numpy(float), enc.transform(g[cat])])
    thr = GERMAN_COST["fp"] / (GERMAN_COST["fp"] + GERMAN_COST["fn"])
    models = {"Decision tree (depth 4)": lambda: DecisionTreeClassifier(max_depth=4, min_samples_leaf=10, random_state=0),
              "Random forest": lambda: RandomForestClassifier(500, min_samples_leaf=3, n_jobs=-1, random_state=0),
              "Gradient boosting": lambda: HistGradientBoostingClassifier(learning_rate=0.05, max_iter=150, random_state=0)}
    res = {k: ([], []) for k in models}
    for t, v in RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=C.SEED).split(X, y):
        for k, mk in models.items():
            p = mk().fit(X[t], y[t]).predict_proba(X[v])[:, 1]
            res[k][0].append(roc_auc_score(y[v], p)); res[k][1].append(expected_cost(y[v], p, thr, GERMAN_COST["fn"], GERMAN_COST["fp"]))
    tab = pd.DataFrame([(k, np.mean(a), np.std(a), np.mean(c), np.std(c)) for k, (a, c) in res.items()],
                       columns=["Model", "AUC", "SD", "Cost at 1/6", "SD "]).set_index("Model")
    save_table("lab4_german", tab, nd=3, col_format="p{6.0cm}cccc")
    num.update({"de_auc_tree": tab.iloc[0]["AUC"], "de_auc_rf": tab.iloc[1]["AUC"], "de_auc_gb": tab.iloc[2]["AUC"],
                "de_cost_rf": tab.iloc[1]["Cost at 1/6"], "de_cost_gb": tab.iloc[2]["Cost at 1/6"]})


# =========================================================================== B
def volatility_trees(num):
    feats = add_targets(market_features(load_market_panel()))
    d = design(feats)
    sp = chrono_split(d.index, horizon=HORIZON)
    tr, va, te = d.loc[sp.train], d.loc[sp.val], d.loc[sp.test]
    trva = pd.concat([tr, va])
    models = {"Linear (Lab 2)": make_model(1),
              "Regression tree (depth 5)": DecisionTreeRegressor(max_depth=5, min_samples_leaf=50, random_state=C.SEED),
              "Random forest": RandomForestRegressor(300, min_samples_leaf=20, max_features=0.5, n_jobs=-1, random_state=C.SEED),
              "Gradient boosting": HistGradientBoostingRegressor(learning_rate=0.05, max_iter=300, min_samples_leaf=50,
                                                                 random_state=C.SEED)}
    rows, preds = [], {}
    for name, m in models.items():
        m.fit(tr[VFEAT], tr[TARGET]); v = rmse(va[TARGET], m.predict(va[VFEAT]))
        m.fit(trva[VFEAT], trva[TARGET]); preds[name] = m.predict(te[VFEAT])
        rows.append((name, v, rmse(te[TARGET], preds[name])))
    tab = pd.DataFrame(rows, columns=["Model", "Valid. RMSE", "Test RMSE"]).set_index("Model")
    save_table("lab4_vol_trees", tab, nd=3, col_format="p{6.0cm}cc")
    # behaviour outside the training range: vary log VIX, other features at their medians
    grid = np.linspace(np.log(9), np.log(120), 200)
    X0 = pd.DataFrame(np.repeat(trva[VFEAT].median().to_numpy()[None], len(grid), 0), columns=VFEAT)
    X0["log_vix"] = grid
    vmax = float(np.exp(trva["log_vix"].max()))
    f, ax = P.fig(2.7)
    for (name, m), col in zip(models.items(), P.CAT):
        ax.plot(np.exp(grid), 100 * np.exp(m.predict(X0)), color=col, lw=1.2, label=name)
    ax.axvline(vmax, color=P.MUTED, ls=":", lw=0.9)
    ax.text(vmax, 0.03, " max VIX of 2000--2018", transform=ax.get_xaxis_transform(), fontsize=9, va="bottom", color=P.INK2)
    ax.set_xscale("log"); ax.set_xlabel("VIX (other features at their medians)"); ax.set_ylabel("predicted volatility, %")
    ax.legend(fontsize=9)
    P.save(f, "lab4_vol_extrapolation")
    num.update({"vt_lin_test": tab.iloc[0]["Test RMSE"], "vt_tree_test": tab.iloc[1]["Test RMSE"],
                "vt_rf_test": tab.iloc[2]["Test RMSE"], "vt_gb_test": tab.iloc[3]["Test RMSE"], "vt_vix_max": vmax})


# =========================================================================== C
def global_boosting(num):
    p = build_panel(5)
    m = split_masks(p)
    feats = PFEAT + ["is_index", "is_asset"]
    p = p.assign(is_index=(p["group"] == "index").astype(float), is_asset=(p["group"] == "asset").astype(float))
    y = p["y_5"].to_numpy()
    X = p[feats]
    # number of iterations chosen on the validation block (staged predictions of a model fitted on train)
    gb = HistGradientBoostingRegressor(learning_rate=0.05, max_iter=800, min_samples_leaf=200, early_stopping=False,
                                       random_state=C.SEED).fit(X[m["train"]], y[m["train"]])
    vl = [rmse(y[m["val"]], pr) for pr in gb.staged_predict(X[m["val"]])]
    n_best = int(np.argmin(vl)) + 1
    trva = m["train"] | m["val"]
    glob = HistGradientBoostingRegressor(learning_rate=0.05, max_iter=n_best, min_samples_leaf=200, early_stopping=False,
                                         random_state=C.SEED).fit(X[trva], y[trva])
    pg = glob.predict(X)
    rows = []
    for s, g in p.groupby("series"):
        idx = g.index.to_numpy()
        tri, tei = idx[trva[idx]], idx[m["test"][idx]]
        if len(tei) < 50 or len(tri) < 500:
            continue
        har = LinearRegression().fit(p.loc[tri, PFEAT], y[tri])
        loc = HistGradientBoostingRegressor(learning_rate=0.05, max_iter=n_best, min_samples_leaf=50,
                                            random_state=C.SEED).fit(p.loc[tri, PFEAT], y[tri])
        rows.append((s, g["group"].iloc[0], len(tri), rmse(y[tei], har.predict(p.loc[tei, PFEAT])),
                     rmse(y[tei], loc.predict(p.loc[tei, PFEAT])), rmse(y[tei], pg[tei])))
    r = pd.DataFrame(rows, columns=["series", "group", "n_train", "har", "local_gb", "global_gb"])
    r.to_csv(C.RESULTS_DIR / "lab4_global_by_series.csv", index=False)
    spx = r[r["series"] == "^GSPC"].iloc[0]
    summ = pd.DataFrame([
        ("Linear HAR model for each series", r["har"].mean(), 1.0, 0, float(spx["har"])),
        ("Gradient boosting for each series", r["local_gb"].mean(), float((r["local_gb"] / r["har"]).mean()),
         int((r["local_gb"] < r["har"]).sum()), float(spx["local_gb"])),
        ("One gradient-boosting model for all series", r["global_gb"].mean(), float((r["global_gb"] / r["har"]).mean()),
         int((r["global_gb"] < r["har"]).sum()), float(spx["global_gb"])),
    ], columns=["Model", "Mean test RMSE", "Mean ratio to HAR", "Series better than HAR", "S\\&P 500 test RMSE"]).set_index("Model")
    save_table("lab4_global", summ, nd={"Mean test RMSE": 3, "Mean ratio to HAR": 3, "Series better than HAR": 0,
                                        "S\\&P 500 test RMSE": 3}, col_format="p{5.4cm}cccc")
    num.update({"gl_series": len(r), "gl_rows_train": int(m["train"].sum()), "gl_n_iter": n_best,
                "gl_better_than_har": int((r["global_gb"] < r["har"]).sum()), "gl_better_than_local": int((r["global_gb"] < r["local_gb"]).sum()),
                "gl_ratio": float((r["global_gb"] / r["har"]).mean()), "loc_ratio": float((r["local_gb"] / r["har"]).mean()),
                "gl_spx": float(spx["global_gb"]), "har_spx": float(spx["har"]), "loc_spx": float(spx["local_gb"])})
    f, ax = P.fig(2.8)
    for k, (g, col) in enumerate(zip(("index", "asset", "industry"), (P.BLUE, P.ORANGE, P.AQUA))):
        rr = r[r["group"] == g]
        for j, (c, mk) in enumerate((("local_gb", "o"), ("global_gb", "s"))):
            v = (rr[c] / rr["har"]).to_numpy()
            ax.scatter(np.full(len(v), 3 * k + j) + np.random.default_rng(j).uniform(-0.25, 0.25, len(v)), v, s=10,
                       marker=mk, color=col, alpha=0.8)
    ax.axhline(1, color=P.INK2, lw=0.7)
    ax.set_xticks([0, 1, 3, 4, 6, 7]); ax.set_xticklabels(["indices\nlocal", "indices\nglobal", "assets\nlocal", "assets\nglobal",
                                                          "industries\nlocal", "industries\nglobal"], fontsize=9)
    ax.set_ylabel("test RMSE relative to HAR")
    P.save(f, "lab4_global")


def main():
    num = {}
    Xtr, Xva, Xte, ytr, yva, yte = credit_split()
    alpha = pruning(num, Xtr, Xva, ytr, yva)
    impurity_and_rules(num, Xtr, Xva, ytr, yva, alpha)
    rf = forest(num, Xtr, Xva, ytr, yva)
    gb = boosting(num, Xtr, Xva, ytr, yva)
    importance(num, rf, gb, Xtr, Xva, ytr, yva)
    compare(num, alpha, rf, gb, Xtr, Xva, Xte, ytr, yva, yte)
    german(num)
    volatility_trees(num)
    global_boosting(num)
    save_numbers("lab4", num)
    print(num)


if __name__ == "__main__":
    main()
