"""Lab 3. Credit-default classification: logistic regression vs perceptrons.

Run:  python -m lab3_classification.run
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.calibration import calibration_curve
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Perceptron as SkPerceptron
from sklearn.metrics import (average_precision_score, brier_score_loss, confusion_matrix,
                             f1_score, precision_recall_curve, precision_score,
                             recall_score, roc_auc_score, roc_curve, accuracy_score,
                             balanced_accuracy_score)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from dmfin import config as C
from dmfin import plotting as P
from dmfin.data import load_credit, load_market_panel
from dmfin.features import add_targets, market_features
from dmfin.reporting import save_numbers, save_table
from dmfin.splits import chrono_split

from .model import LogisticNeuron, Perceptron, best_threshold, expected_cost, prepare

C_FN, C_FP = 5.0, 1.0


def lr(Cval=1.0, cw=None, max_iter=2000):
    return Pipeline([("scale", StandardScaler()),
                     ("lr", LogisticRegression(C=Cval, class_weight=cw, max_iter=max_iter))])


def scores_table(y, s, thr, prob=True):
    pred = (s >= thr).astype(int)
    return {"Accuracy": accuracy_score(y, pred), "Bal. acc.": balanced_accuracy_score(y, pred),
            "Precision": precision_score(y, pred, zero_division=0), "Recall": recall_score(y, pred),
            "F1": f1_score(y, pred), "ROC AUC": roc_auc_score(y, s),
            "PR AUC": average_precision_score(y, s),
            "Cost": expected_cost(y, s, thr, C_FN, C_FP),
            "Brier": brier_score_loss(y, s) if prob else np.nan}


def main():
    num = {}
    X, y = prepare(load_credit())
    Xtrva, Xte, ytrva, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=C.SEED)
    Xtr, Xva, ytr, yva = train_test_split(Xtrva, ytrva, test_size=0.25, stratify=ytrva, random_state=C.SEED)
    num.update({"n_train": len(ytr), "n_val": len(yva), "n_test": len(yte), "p": X.shape[1],
                "rate_train": 100 * ytr.mean(), "rate_val": 100 * yva.mean(), "rate_test": 100 * yte.mean(),
                "bayes_thr": C_FP / (C_FP + C_FN)})
    feat_names = list(X.columns)

    # ----------------------------------------------------------- EDA figure
    raw = load_credit()
    f, axs = P.fig(2.6, ncols=2)
    g = raw.groupby(raw["PAY_1"].clip(-2, 4))["default"].agg(["mean", "size"])
    axs[0].bar(g.index.astype(int), 100 * g["mean"], color=P.BLUE, width=0.7)
    for k, (m, n) in g.iterrows():
        axs[0].text(k, 100 * m + 2, f"{int(n)}", ha="center", fontsize=9, color=P.INK2)
    axs[0].set_xlabel("repayment status PAY_1 (months late)")
    axs[0].set_ylabel("default rate, %")
    dec = pd.qcut(raw["LIMIT_BAL"], 10, duplicates="drop")
    g2 = raw.groupby(dec, observed=True)["default"].mean()
    axs[1].bar(range(1, len(g2) + 1), 100 * g2.values, color=P.BLUE, width=0.7)
    axs[1].set_xlabel("credit-limit decile"); axs[1].set_ylabel("default rate, %")
    axs[1].set_xticks(range(1, len(g2) + 1))
    f.tight_layout(); P.save(f, "lab3_eda")
    num.update({"rate_pay_ontime": 100 * raw.loc[raw.PAY_1 <= 0, "default"].mean(),
                "rate_pay_2": 100 * raw.loc[raw.PAY_1 == 2, "default"].mean(),
                "rate_lim_d1": 100 * g2.iloc[0], "rate_lim_d10": 100 * g2.iloc[-1]})

    # ------------------------------------------- why normalisation matters
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        raw_lr = LogisticRegression(max_iter=100).fit(Xtr, ytr)
        raw_lr_long = LogisticRegression(max_iter=5000).fit(Xtr, ytr)
    std_lr = lr().fit(Xtr, ytr)
    num.update({"iter_raw": int(raw_lr_long.n_iter_[0]), "iter_std": int(std_lr.named_steps["lr"].n_iter_[0]),
                "auc_raw100": roc_auc_score(yva, raw_lr.predict_proba(Xva)[:, 1]),
                "auc_std": roc_auc_score(yva, std_lr.predict_proba(Xva)[:, 1])})
    pr_raw = Perceptron(epochs=10, average=True).fit(Xtr.to_numpy(), ytr.to_numpy())
    num["auc_perc_raw"] = roc_auc_score(yva, pr_raw.decision_function(Xva.to_numpy()))

    sc = StandardScaler().fit(Xtr)
    Ztr, Zva, Zte = sc.transform(Xtr), sc.transform(Xva), sc.transform(Xte)

    # --------------------------------------- logistic regression: tune C by CV
    skf = StratifiedKFold(5, shuffle=True, random_state=C.SEED)
    grid = [1e-4, 1e-3, 1e-2, 1e-1, 1, 10]
    cvres = []
    for Cv in grid:
        aucs = []
        for a, b in skf.split(Xtr, ytr):
            m = lr(Cv).fit(Xtr.iloc[a], ytr.iloc[a])
            aucs.append(roc_auc_score(ytr.iloc[b], m.predict_proba(Xtr.iloc[b])[:, 1]))
        cvres.append((Cv, np.mean(aucs), np.std(aucs) / np.sqrt(5)))
    cvdf = pd.DataFrame(cvres, columns=["C", "auc", "se"])
    C_best = float(cvdf.loc[cvdf.auc.idxmax(), "C"])
    num.update({"C_best": f"{C_best:g}", "cv_auc_best": cvdf.auc.max(),
                "cv_auc_min": cvdf.auc.min(), "C_min": f"{cvdf.loc[cvdf.auc.idxmin(), 'C']:g}"})
    save_table("lab3_cv_C", cvdf.rename(columns={"auc": "CV ROC AUC", "se": "SE"}).set_index("C"), nd=4)

    models = {}
    m_lr = lr(C_best).fit(Xtr, ytr); models["Logistic regression"] = ("p", m_lr.predict_proba)
    m_lrb = lr(C_best, "balanced").fit(Xtr, ytr); models["Logistic regression, balanced"] = ("p", m_lrb.predict_proba)

    # -------------------------------------------------- statsmodels inference
    keep = ["log_limit", "age", "female", "married", "pay_1", "pay_2", "n_delinquent",
            "max_delay", "utilisation", "pay_ratio", "pay_amt1_", "bill_amt1_"]
    keep = [k for k in keep if k in feat_names] + ["pay_amt1", "bill_amt1"]
    keep = list(dict.fromkeys([k for k in keep if k in feat_names]))
    Zk = pd.DataFrame(Ztr, columns=feat_names, index=Xtr.index)[keep]
    logit = sm.Logit(ytr.to_numpy(), sm.add_constant(Zk.to_numpy())).fit(disp=0)
    ci = logit.conf_int()
    otab = pd.DataFrame({"$\\hat\\beta$": logit.params, "SE": logit.bse, "$z$": logit.tvalues,
                         "Odds ratio": np.exp(logit.params), "OR 95\\% CI": [
                             f"{np.exp(a):.2f}--{np.exp(b):.2f}" for a, b in ci]},
                        index=["const"] + keep)
    otab.index.name = "Feature (standardized)"
    save_table("lab3_logit", otab, nd=3, col_format="lccccc")
    num.update({"or_pay1": float(np.exp(logit.params[1 + keep.index("pay_1")])),
                "or_limit": float(np.exp(logit.params[1 + keep.index("log_limit")])),
                "pseudo_r2": logit.prsquared})

    # ---------------------------------------------------------- perceptrons
    variants = {"plain": dict(), "pocket": dict(pocket=True), "averaged": dict(average=True)}
    hist = {}
    for name, kw in variants.items():
        p = Perceptron(epochs=30, seed=C.SEED, **kw).fit(Ztr, ytr.to_numpy())
        hist[name] = p.history_
        models[f"Perceptron, {name}"] = ("s", p.decision_function)
    pb = Perceptron(epochs=30, average=True, class_weight="balanced", seed=C.SEED).fit(Ztr, ytr.to_numpy())
    models["Perceptron, averaged, balanced"] = ("s", pb.decision_function)
    skp = SkPerceptron(max_iter=50, random_state=C.SEED).fit(Ztr, ytr)
    models["Perceptron (scikit-learn)"] = ("s", skp.decision_function)

    f, ax = P.fig(2.6)
    h = np.array(hist["plain"])
    for j, name, col in [(1, "last iterate (plain)", P.RED), (2, "pocket weights", P.BLUE), (3, "averaged weights", P.AQUA)]:
        ax.plot(np.arange(1, len(h) + 1), 100 * h[:, j], "o-", ms=3, color=col, label=name)
    ax.axhline(100 * ytr.mean(), color=P.INK2, ls="--", lw=0.8, label="always predict 'no default'")
    ax.set_xlabel("epoch"); ax.set_ylabel("training error after the epoch, %")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1), ncol=4)
    P.save(f, "lab3_perceptron_epochs")
    hp = np.array(hist["plain"])
    num.update({"perc_plain_err_min": 100 * hp[:, 1].min(), "perc_plain_err_max": 100 * hp[:, 1].max(),
                "perc_mistakes_last": int(hp[-1, 0]), "perc_pocket_err": 100 * hp[-1, 2],
                "perc_avg_err": 100 * hp[-1, 3]})

    # ------------------------------------------------- logistic neuron by SGD
    ln = LogisticNeuron(lr=0.2, epochs=60, l2=1e-4, seed=C.SEED).fit(Ztr, ytr.to_numpy(), Zva, yva.to_numpy())
    models["Single sigmoid neuron (SGD)"] = ("p", ln.predict_proba)
    w_lr = m_lr.named_steps["lr"].coef_.ravel()
    num.update({"neuron_lr_coef_corr": float(np.corrcoef(w_lr, ln.w_)[0, 1]),
                "neuron_final_val_loss": ln.history_[-1][1]})
    f, ax = P.fig(2.4)
    h = np.array(ln.history_)
    ax.plot(np.arange(1, len(h) + 1), h[:, 0], color=P.BLUE, label="training")
    ax.plot(np.arange(1, len(h) + 1), h[:, 1], color=P.ORANGE, label="validation")
    ax.set_xlabel("epoch"); ax.set_ylabel("cross-entropy"); ax.legend()
    P.save(f, "lab3_neuron_loss")

    # ----------------------------------------------------------------- PCA
    pca = PCA().fit(Ztr)
    evr = np.cumsum(pca.explained_variance_ratio_)
    ks = list(range(1, len(evr) + 1))
    auc_k = []
    for k in ks:
        m = Pipeline([("s", StandardScaler()), ("pca", PCA(k)), ("lr", LogisticRegression(C=C_best, max_iter=2000))])
        m.fit(Xtr, ytr); auc_k.append(roc_auc_score(yva, m.predict_proba(Xva)[:, 1]))
    k95 = int(np.searchsorted(evr, 0.95) + 1)
    f, ax = P.fig(2.6, ncols=2)
    ax[0].plot(ks, 100 * evr, "o-", ms=3, color=P.BLUE)
    ax[0].axhline(95, color=P.INK2, ls="--", lw=0.8); ax[0].axvline(k95, color=P.INK2, ls="--", lw=0.8)
    ax[0].set_xlabel("number of components $k$"); ax[0].set_ylabel("cumulative variance, %")
    ax[1].plot(ks, auc_k, "o-", ms=3, color=P.ORANGE)
    ax[1].axhline(roc_auc_score(yva, m_lr.predict_proba(Xva)[:, 1]), color=P.INK2, ls="--", lw=0.8)
    ax[1].set_xlabel("number of components $k$"); ax[1].set_ylabel("validation ROC AUC")
    f.tight_layout(); P.save(f, "lab3_pca")
    k_auc = int(ks[int(np.argmax(np.array(auc_k) >= max(auc_k) - 0.002))])
    mp = Pipeline([("s", StandardScaler()), ("pca", PCA(k95)), ("lr", LogisticRegression(C=C_best, max_iter=2000))]).fit(Xtr, ytr)
    models[f"PCA({k95}) + logistic regression"] = ("p", mp.predict_proba)
    num.update({"k95": k95, "k_auc": k_auc, "pc1_var": 100 * pca.explained_variance_ratio_[0],
                "auc_pca_k95": auc_k[k95 - 1], "auc_pca_k1": auc_k[0]})

    # ------------------------------------------------------- tree-based models
    from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
    from sklearn.inspection import permutation_importance
    from sklearn.tree import DecisionTreeClassifier, plot_tree
    path = DecisionTreeClassifier(random_state=C.SEED, min_samples_leaf=20).cost_complexity_pruning_path(Ztr, ytr)
    alphas = np.unique(np.quantile(path.ccp_alphas[:-1], np.linspace(0, 1, 25)))
    cv_tree = []
    for a in alphas:
        aucs = [roc_auc_score(ytr.iloc[b], DecisionTreeClassifier(ccp_alpha=a, min_samples_leaf=20, random_state=C.SEED)
                              .fit(Ztr[t], ytr.iloc[t]).predict_proba(Ztr[b])[:, 1]) for t, b in skf.split(Ztr, ytr)]
        cv_tree.append(np.mean(aucs))
    a_best = float(alphas[int(np.argmax(cv_tree))])
    tree = DecisionTreeClassifier(ccp_alpha=a_best, min_samples_leaf=20, random_state=C.SEED).fit(Ztr, ytr)
    models["Decision tree (CART, pruned)"] = ("p", tree.predict_proba)
    rf = RandomForestClassifier(n_estimators=500, min_samples_leaf=5, max_features="sqrt", n_jobs=-1,
                                random_state=C.SEED, oob_score=True).fit(Ztr, ytr)
    models["Random forest (500 trees)"] = ("p", rf.predict_proba)
    gb = HistGradientBoostingClassifier(learning_rate=0.05, max_iter=500, early_stopping=True,
                                        validation_fraction=0.15, random_state=C.SEED).fit(Ztr, ytr)
    models["Gradient boosting"] = ("p", gb.predict_proba)
    small = DecisionTreeClassifier(max_depth=3, min_samples_leaf=200, random_state=C.SEED).fit(Xtr, ytr)
    pim = permutation_importance(rf, Zva, yva, scoring="roc_auc", n_repeats=5, random_state=C.SEED, n_jobs=-1)
    imp = pd.Series(pim.importances_mean, index=feat_names).sort_values(ascending=False).head(10)
    # the tree gets the full width (eight leaves side by side), the importances are drawn below it
    f, axs = P.fig(5.4, nrows=2, gridspec_kw={"height_ratios": [1.5, 1]})
    plot_tree(small, feature_names=feat_names, class_names=["no", "default"], filled=False, impurity=False,
              proportion=True, precision=2, fontsize=9, label="root", ax=axs[0])
    axs[0].grid(False)
    axs[1].barh(range(len(imp)), imp.values, color=P.BLUE)
    axs[1].set_yticks(range(len(imp))); axs[1].set_yticklabels(imp.index, fontsize=9); axs[1].invert_yaxis()
    axs[1].set_xlabel("drop of validation AUC")
    f.tight_layout(); P.save(f, "lab3_trees")
    num.update({"tree_alpha": f"{a_best:.1e}".replace("e-0", "\\cdot 10^{-") + "}", "tree_leaves": int(tree.get_n_leaves()),
                "tree_depth": int(tree.get_depth()), "rf_oob": rf.oob_score_, "gb_iter": int(gb.n_iter_),
                "imp_top": imp.index[0].replace("_", "\\_"), "imp_top_val": float(imp.iloc[0]),
                "imp_second": imp.index[1].replace("_", "\\_")})

    # ------------------------------------------- threshold choice on validation
    def score_fn(kind, fn, Z, Xdf):
        inp = Xdf if fn.__self__.__class__.__name__ == "Pipeline" else Z
        out = fn(inp)
        return out[:, 1] if kind == "p" else out

    rows, curves = [], {}
    for name, (kind, fn) in models.items():
        sv = score_fn(kind, fn, Zva, Xva); st = score_fn(kind, fn, Zte, Xte)
        thr = 0.5 if kind == "p" else 0.0
        thr_c, _, _ = best_threshold(yva.to_numpy(), sv, C_FN, C_FP)
        r05 = scores_table(yte.to_numpy(), st, thr, prob=(kind == "p"))
        rc = scores_table(yte.to_numpy(), st, thr_c, prob=(kind == "p"))
        rows.append((name, r05["Accuracy"], r05["Recall"], r05["F1"], rc["Recall"], rc["Precision"],
                     rc["Cost"], r05["ROC AUC"], r05["PR AUC"], r05["Brier"]))
        curves[name] = (sv, st, thr_c)
    res = pd.DataFrame(rows, columns=["Model", "Acc.", "Rec.", "F1", "Rec.*", "Prec.*", "Cost*",
                                      "AUC", "PR AUC", "Brier"]).set_index("Model")
    save_table("lab3_results", res, nd=3, col_format="p{4.3cm}ccccccccc")
    print(res.round(3).to_string())
    num.update({"lr_acc": res.loc["Logistic regression", "Acc."], "lr_auc": res.loc["Logistic regression", "AUC"],
                "lr_rec05": res.loc["Logistic regression", "Rec."], "lr_rec_c": res.loc["Logistic regression", "Rec.*"],
                "lr_prec_c": res.loc["Logistic regression", "Prec.*"], "lr_cost": res.loc["Logistic regression", "Cost*"],
                "lr_brier": res.loc["Logistic regression", "Brier"], "lr_prauc": res.loc["Logistic regression", "PR AUC"],
                "lrb_brier": res.loc["Logistic regression, balanced", "Brier"],
                "lrb_auc": res.loc["Logistic regression, balanced", "AUC"],
                "perc_plain_auc": res.loc["Perceptron, plain", "AUC"],
                "perc_avg_auc": res.loc["Perceptron, averaged", "AUC"],
                "perc_avg_acc": res.loc["Perceptron, averaged", "Acc."],
                "perc_pocket_auc": res.loc["Perceptron, pocket", "AUC"],
                "perc_sk_auc": res.loc["Perceptron (scikit-learn)", "AUC"],
                "neuron_auc": res.loc["Single sigmoid neuron (SGD)", "AUC"],
                "pca_auc": res.loc[f"PCA({k95}) + logistic regression", "AUC"],
                "tree_auc": res.loc["Decision tree (CART, pruned)", "AUC"],
                "rf_auc": res.loc["Random forest (500 trees)", "AUC"], "gb_auc": res.loc["Gradient boosting", "AUC"],
                "rf_brier": res.loc["Random forest (500 trees)", "Brier"], "gb_cost": res.loc["Gradient boosting", "Cost*"],
                "gb_brier": res.loc["Gradient boosting", "Brier"],
                "allno_cost": C_FN * yte.mean(), "majority_acc": 1 - yte.mean(),
                "lr_thr_c": curves["Logistic regression"][2]})

    # ------------------------------------------------- ROC and PR curves (test)
    show = ["Logistic regression", "Single sigmoid neuron (SGD)", "Perceptron, averaged", "Perceptron, plain"]
    cols = [P.BLUE, P.AQUA, P.ORANGE, P.RED]
    f, axs = P.fig(3.0, ncols=2)
    for name, col in zip(show, cols):
        st = curves[name][1]
        fpr, tpr, _ = roc_curve(yte, st)
        axs[0].plot(fpr, tpr, color=col, lw=1.1, label=f"{name} ({roc_auc_score(yte, st):.3f})")
        pr, rc, _ = precision_recall_curve(yte, st)
        axs[1].plot(rc, pr, color=col, lw=1.1)
    axs[0].plot([0, 1], [0, 1], color=P.MUTED, ls="--", lw=0.8)
    axs[0].set_xlabel("false positive rate"); axs[0].set_ylabel("true positive rate (recall)")
    axs[0].legend(fontsize=9, loc="lower right")
    axs[1].axhline(yte.mean(), color=P.MUTED, ls="--", lw=0.8)
    axs[1].set_xlabel("recall"); axs[1].set_ylabel("precision")
    f.tight_layout(); P.save(f, "lab3_roc_pr")

    # ------------------------------------------ calibration & threshold-cost
    f, axs = P.fig(2.8, ncols=2)
    for name, col in [("Logistic regression", P.BLUE), ("Logistic regression, balanced", P.ORANGE)]:
        st = curves[name][1]
        fr, mp_ = calibration_curve(yte, st, n_bins=10, strategy="quantile")
        axs[0].plot(mp_, fr, "o-", ms=3, color=col, label=name.replace("Logistic regression", "LR"))
    axs[0].plot([0, 1], [0, 1], color=P.MUTED, ls="--", lw=0.8)
    axs[0].set_xlabel("mean predicted probability"); axs[0].set_ylabel("observed default rate")
    axs[0].legend(fontsize=9, loc="upper left")
    sv = curves["Logistic regression"][0]
    thr_c, grid_t, costs = best_threshold(yva.to_numpy(), sv, C_FN, C_FP, grid=np.linspace(0.02, 0.9, 177))
    axs[1].plot(grid_t, costs, color=P.BLUE)
    axs[1].axvline(thr_c, color=P.ORANGE, lw=0.9, label=f"validation optimum {thr_c:.2f}")
    axs[1].axvline(C_FP / (C_FP + C_FN), color=P.AQUA, lw=0.9, ls="--", label=f"Bayes rule $1/6$")
    axs[1].axvline(0.5, color=P.MUTED, lw=0.9, ls=":", label="default 0.5")
    axs[1].set_xlabel("decision threshold"); axs[1].set_ylabel("expected cost per client")
    axs[1].legend(fontsize=9, loc="lower right", handlelength=1.2, borderaxespad=0.1)
    f.tight_layout(); P.save(f, "lab3_calibration_cost")

    # --------------------------------------------------- confusion matrices
    st = curves["Logistic regression"][1]
    f, axs = P.fig(2.5, ncols=2)
    for ax, thr, ttl in [(axs[0], 0.5, "threshold 0.50"), (axs[1], curves["Logistic regression"][2],
                                                           f"cost-optimal threshold {curves['Logistic regression'][2]:.2f}")]:
        cm = confusion_matrix(yte, (st >= thr).astype(int))
        ax.imshow(cm, cmap=P.SEQ_CELL, vmin=0, vmax=cm.max())
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]}", ha="center", va="center", fontsize=10, color=P.INK)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["pred. no", "pred. yes"])
        ax.set_yticks([0, 1]); ax.set_yticklabels(["true no", "true yes"])
        ax.set_title(ttl, fontsize=9); ax.grid(False)
        if thr == 0.5:
            num.update({"cm05_tn": int(cm[0, 0]), "cm05_fp": int(cm[0, 1]), "cm05_fn": int(cm[1, 0]), "cm05_tp": int(cm[1, 1])})
        else:
            num.update({"cmc_tn": int(cm[0, 0]), "cmc_fp": int(cm[0, 1]), "cmc_fn": int(cm[1, 0]), "cmc_tp": int(cm[1, 1])})
    f.tight_layout(); P.save(f, "lab3_confusion")

    # ------------------------------------------------------- learning curves
    sizes = [250, 500, 1000, 2000, 4000, 8000, len(ytr)]
    lc = []
    rng = np.random.default_rng(C.SEED)
    for n in sizes:
        a1, a2 = [], []
        for rep in range(5):
            idx = rng.choice(len(ytr), size=n, replace=False)
            m = lr(C_best).fit(Xtr.iloc[idx], ytr.iloc[idx])
            a1.append(roc_auc_score(yva, m.predict_proba(Xva)[:, 1]))
            p = Perceptron(epochs=30, average=True, seed=rep).fit(Ztr[idx], ytr.to_numpy()[idx])
            a2.append(roc_auc_score(yva, p.decision_function(Zva)))
        lc.append((n, np.mean(a1), np.std(a1), np.mean(a2), np.std(a2)))
    lc = np.array(lc)
    f, ax = P.fig(2.5)
    ax.errorbar(lc[:, 0], lc[:, 1], yerr=lc[:, 2], fmt="o-", ms=3, capsize=2, color=P.BLUE, label="logistic regression")
    ax.errorbar(lc[:, 0], lc[:, 3], yerr=lc[:, 4], fmt="s-", ms=3, capsize=2, color=P.ORANGE, label="averaged perceptron")
    ax.set_xscale("log"); ax.set_xlabel("training-set size"); ax.set_ylabel("validation ROC AUC"); ax.legend()
    P.save(f, "lab3_learning_curve")
    num.update({"lc_lr_250": lc[0, 1], "lc_lr_full": lc[-1, 1], "lc_p_250": lc[0, 3], "lc_p_full": lc[-1, 3]})

    # ------------------------------------------------ bootstrap CI of test AUC
    st = curves["Logistic regression"][1]; yt = yte.to_numpy()
    boots = []
    for _ in range(1000):
        idx = rng.integers(0, len(yt), len(yt))
        boots.append(roc_auc_score(yt[idx], st[idx]))
    num.update({"auc_ci_lo": np.quantile(boots, 0.025), "auc_ci_hi": np.quantile(boots, 0.975)})
    sp = curves["Perceptron, averaged"][1]
    diff = []
    for _ in range(1000):
        idx = rng.integers(0, len(yt), len(yt))
        diff.append(roc_auc_score(yt[idx], st[idx]) - roc_auc_score(yt[idx], sp[idx]))
    num.update({"auc_diff_lo": np.quantile(diff, 0.025), "auc_diff_hi": np.quantile(diff, 0.975)})

    # ----------------------------------- cautionary case: market direction
    fe = add_targets(market_features(load_market_panel()), horizons=(1, 5))
    mk = pd.DataFrame(index=fe.index)
    for k in range(1, 6):
        mk[f"lag{k}"] = fe["ret"].shift(k - 1)
    mk["volume_z"] = fe["volume_z"]; mk["vix"] = np.log(fe["vix"])
    mk["up1"] = (fe["y_ret_1"] > 0).astype(float).where(fe["y_ret_1"].notna())
    mk["up5"] = (fe["y_ret_5"] > 0).astype(float).where(fe["y_ret_5"].notna())
    mk = mk.dropna()
    mrows = []
    for h, tgt in [(1, "up1"), (5, "up5")]:
        spl = chrono_split(mk.index, horizon=h)
        trm = mk.loc[spl.train.append(spl.val)]; tem = mk.loc[spl.test]
        cols_ = [c for c in mk.columns if not c.startswith("up")]
        m = lr(1.0).fit(trm[cols_], trm[tgt])
        s_ = m.predict_proba(tem[cols_])[:, 1]
        mrows.append((f"{h} day" + ("s" if h > 1 else ""), 100 * tem[tgt].mean(), 100 * accuracy_score(tem[tgt], s_ >= 0.5),
                      100 * max(tem[tgt].mean(), 1 - tem[tgt].mean()), roc_auc_score(tem[tgt], s_)))
    mres = pd.DataFrame(mrows, columns=["Horizon", "Up days in test, %", "LR accuracy, %",
                                       "Majority-class accuracy, %", "ROC AUC"]).set_index("Horizon")
    save_table("lab3_market_direction", mres, nd={"Up days in test, %": 1, "LR accuracy, %": 1,
                                                  "Majority-class accuracy, %": 1, "ROC AUC": 3})
    num.update({"mkt_acc1": mres.iloc[0, 1], "mkt_base1": mres.iloc[0, 2], "mkt_auc1": mres.iloc[0, 3],
                "mkt_acc5": mres.iloc[1, 1], "mkt_base5": mres.iloc[1, 2], "mkt_auc5": mres.iloc[1, 3]})

    save_numbers("lab3", num)
    print(num)


if __name__ == "__main__":
    main()
