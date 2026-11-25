"""Lab 3 (iteration 2). Multilayer perceptrons and further classifiers.

A. A study of the multilayer perceptron on the credit data (Lab 3 split): topology,
   regularization, input scaling, optimizer and learning rate, seeds, learning curves
   against logistic regression and gradient boosting, calibration.
B. Further classifiers: linear and RBF support vector machines, k nearest neighbours,
   linear discriminant analysis, Gaussian naive Bayes.
C. The German credit data (1 000 applicants) with its official cost matrix,
   evaluated by repeated stratified cross-validation.

Run:  python -m lab3_classification.run_ext
"""
from __future__ import annotations

import warnings
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import torch
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.frozen import FrozenEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC, LinearSVC

from dmfin import config as C
from dmfin import plotting as P
from dmfin.data import load_credit
from dmfin.data_ext import GERMAN_COST, load_german_credit
from dmfin.reporting import save_numbers, save_table

from .mlp import fit_mlp, predict_proba
from .model import best_threshold, expected_cost, prepare

C_FN, C_FP = 5.0, 1.0
WIDTHS, DEPTHS = [8, 32, 128, 512], [1, 2, 3]


def split():
    X, y = prepare(load_credit())
    Xtrva, Xte, ytrva, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=C.SEED)
    Xtr, Xva, ytr, yva = train_test_split(Xtrva, ytrva, test_size=0.25, stratify=ytrva, random_state=C.SEED)
    sc = StandardScaler().fit(Xtr)
    return (Xtr, Xva, Xte, ytr.to_numpy(), yva.to_numpy(), yte.to_numpy(),
            sc.transform(Xtr), sc.transform(Xva), sc.transform(Xte))


def _job(args):
    """One MLP fit; returns validation AUC, history and test probabilities."""
    torch.set_num_threads(1)
    key, cfg, scaled, n_sub, sub_seed = args
    Xtr, Xva, Xte, ytr, yva, yte, Ztr, Zva, Zte = split()
    A, B, T = (Ztr, Zva, Zte) if scaled else (Xtr.to_numpy(), Xva.to_numpy(), Xte.to_numpy())
    if n_sub:
        idx = np.random.default_rng(sub_seed).choice(len(A), n_sub, replace=False)
        A, ytr = A[idx], ytr[idx]
    m, hist = fit_mlp(A, ytr, B, yva, **cfg)
    pv, pt = predict_proba(m, B), predict_proba(m, T)
    auc = roc_auc_score(yva, pv) if np.all(np.isfinite(pv)) else np.nan
    return key, auc, hist, pv, pt


def run_jobs(jobs, workers=12):
    with ProcessPoolExecutor(workers) as ex:
        return {r[0]: r[1:] for r in ex.map(_job, jobs)}


def metrics(y, p, thr, prob=True):
    return {"AUC": roc_auc_score(y, p), "PR AUC": average_precision_score(y, p),
            "Brier": brier_score_loss(y, p) if prob else np.nan, "Cost": expected_cost(y, p, thr, C_FN, C_FP)}


def part_a(num):
    Xtr, Xva, Xte, ytr, yva, yte, Ztr, Zva, Zte = split()
    base = dict(dropout=0.1, weight_decay=1e-4, lr=1e-3)
    jobs = []
    for d in DEPTHS:                                             # 1. topology
        for w in WIDTHS:
            for s in range(3):
                jobs.append((("topo", d, w, s), dict(base, width=w, depth=d, seed=s), True, None, None))
    big = dict(width=256, depth=2, lr=1e-3, max_epochs=100)       # 2. regularization of a large network
    regs = {"none (100 epochs)": dict(big, early_stopping=False),
            "early stopping": dict(big),
            "weight decay $10^{-3}$": dict(big, weight_decay=1e-3, early_stopping=False),
            "dropout 0.3": dict(big, dropout=0.3, early_stopping=False),
            "all three": dict(big, dropout=0.3, weight_decay=1e-3)}
    for k, cfg in regs.items():
        jobs.append((("reg", k), dict(cfg, seed=0), True, None, None))
    for scaled in (False, True):                                  # 3. input scaling
        jobs.append((("scale", scaled), dict(base, width=64, depth=2, seed=0, max_epochs=60, early_stopping=False),
                     scaled, None, None))
    opts = {"SGD, lr 0.01": dict(optimizer="sgd", lr=0.01), "SGD, lr 0.1": dict(optimizer="sgd", lr=0.1),
            "AdamW, lr $10^{-4}$": dict(lr=1e-4), "AdamW, lr $10^{-3}$": dict(lr=1e-3), "AdamW, lr $10^{-2}$": dict(lr=1e-2)}
    for k, o in opts.items():                                     # 4. optimizer and learning rate
        jobs.append((("opt", k), dict(width=64, depth=2, dropout=0.1, weight_decay=1e-4, seed=0, max_epochs=60,
                                      early_stopping=False, **o), True, None, None))
    for s in range(10):                                           # 5. seeds of the reference network
        jobs.append((("seed", s), dict(base, width=64, depth=2, seed=s), True, None, None))
    sizes = [250, 500, 1000, 2000, 5000, 10000]                   # 6. learning curve
    for n in sizes:
        for s in range(3):
            jobs.append((("lc", n, s), dict(base, width=64, depth=2, seed=s), True, n, s))
    R = run_jobs(jobs)

    # ---- 1. topology heat map
    auc = np.array([[np.mean([R[("topo", d, w, s)][0] for s in range(3)]) for w in WIDTHS] for d in DEPTHS])
    bd, bw = np.unravel_index(np.nanargmax(auc), auc.shape)
    num.update({"mlp_best_depth": DEPTHS[bd], "mlp_best_width": WIDTHS[bw], "mlp_best_auc": float(auc[bd, bw]),
                "mlp_worst_auc": float(np.nanmin(auc)), "mlp_d1w8_auc": float(auc[0, 0])})
    f, axs = P.fig(2.8, ncols=2, gridspec_kw={"width_ratios": [1, 1.3]})
    axs[0].imshow(np.zeros_like(auc), cmap="Greys", vmin=0, vmax=1, aspect="auto"); axs[0].grid(False)   # white cells: a table
    axs[0].set_xticks(np.arange(len(WIDTHS)) - 0.5, minor=True); axs[0].set_yticks(np.arange(len(DEPTHS)) - 0.5, minor=True)
    axs[0].grid(which="minor", color=P.GRID); axs[0].tick_params(which="minor", length=0)
    axs[0].set_xticks(range(len(WIDTHS))); axs[0].set_xticklabels(WIDTHS); axs[0].set_xlabel("width")
    axs[0].set_yticks(range(len(DEPTHS))); axs[0].set_yticklabels(DEPTHS); axs[0].set_ylabel("depth (hidden layers)")
    for i in range(len(DEPTHS)):
        for j in range(len(WIDTHS)):
            axs[0].text(j, i, f"{auc[i, j]:.3f}", ha="center", va="center", fontsize=9, color=P.INK)
    axs[0].set_title("validation AUC (3 seeds)", fontsize=9)
    for k, col in zip(regs, P.CAT):
        h = R[("reg", k)][1]
        axs[1].plot(np.arange(1, len(h) + 1), h[:, 1], color=col, lw=1.1, label=k)
        axs[1].plot(np.arange(1, len(h) + 1), h[:, 0], color=col, lw=0.6, ls=":")
    axs[1].set_ylim(0.38, 0.5); axs[1].set_xlabel("epoch"); axs[1].set_ylabel("log-loss (solid: validation, dotted: training)")
    axs[1].legend(fontsize=9)
    f.tight_layout(); P.save(f, "lab3_mlp_topology")
    regrows = []
    for k in regs:
        auc_k, h, pv, pt = R[("reg", k)]
        regrows.append((k, len(h), float(h[-1, 0]), float(h[:, 1].min()), float(h[-1, 1]), auc_k))
    rt = pd.DataFrame(regrows, columns=["Regularization", "Epochs", "Final train loss", "Best valid. loss",
                                        "Final valid. loss", "Valid. AUC"]).set_index("Regularization")
    save_table("lab3_mlp_regularization", rt, nd={"Epochs": 0, "Final train loss": 3, "Best valid. loss": 3,
                                                  "Final valid. loss": 3, "Valid. AUC": 3}, col_format="p{4.2cm}ccccc")
    num.update({"reg_none_gap": float(rt.iloc[0]["Final valid. loss"] - rt.iloc[0]["Best valid. loss"]),
                "reg_none_auc": float(rt.iloc[0]["Valid. AUC"]), "reg_all_auc": float(rt.iloc[-1]["Valid. AUC"])})

    # ---- 3., 4. scaling and optimizers
    f, axs = P.fig(2.6, ncols=2)
    for scaled, col, name in ((False, P.RED, "raw inputs"), (True, P.BLUE, "standardized inputs")):
        h = R[("scale", scaled)][1]
        axs[0].plot(np.arange(1, len(h) + 1), np.clip(h[:, 1], None, 2.0), color=col, label=name)
    axs[0].set_xlabel("epoch"); axs[0].set_ylabel("validation log-loss"); axs[0].set_yscale("log"); axs[0].legend(fontsize=9)
    for k, col in zip(opts, P.CAT):
        h = R[("opt", k)][1]
        axs[1].plot(np.arange(1, len(h) + 1), np.clip(h[:, 1], None, 0.6), color=col, lw=1, label=k)
    axs[1].set_xlabel("epoch"); axs[1].set_ylabel("validation log-loss"); axs[1].legend(fontsize=9)
    f.tight_layout(); P.save(f, "lab3_mlp_scaling_opt")
    num.update({"scale_raw_auc": R[("scale", False)][0], "scale_std_auc": R[("scale", True)][0],
                **{f"opt_{i}_auc": R[("opt", k)][0] for i, k in enumerate(opts)}})

    # ---- 5. seeds and the reference network on the test block
    aucs = [R[("seed", s)][0] for s in range(10)]
    pv_ens = np.mean([R[("seed", s)][2] for s in range(10)], axis=0)
    pt_ens = np.mean([R[("seed", s)][3] for s in range(10)], axis=0)
    num.update({"seed_auc_mean": float(np.mean(aucs)), "seed_auc_sd": float(np.std(aucs)),
                "seed_auc_min": float(np.min(aucs)), "seed_auc_max": float(np.max(aucs)),
                "ens_auc_val": roc_auc_score(yva, pv_ens)})

    # ---- 6. learning curves: MLP vs logistic regression vs gradient boosting
    lc = {"MLP": [], "Logistic regression": [], "Gradient boosting": []}
    for n in sizes + [len(ytr)]:
        a = {k: [] for k in lc}
        for s in range(3 if n < len(ytr) else 1):
            if n < len(ytr):
                a["MLP"].append(R[("lc", n, s)][0])
                idx = np.random.default_rng(s).choice(len(Ztr), n, replace=False)
            else:
                a["MLP"].append(np.mean(aucs)); idx = np.arange(len(Ztr))
            lr = LogisticRegression(C=10, max_iter=3000).fit(Ztr[idx], ytr[idx])
            gb = HistGradientBoostingClassifier(random_state=s, early_stopping=False, max_iter=150, learning_rate=0.05).fit(Xtr.iloc[idx], ytr[idx])
            a["Logistic regression"].append(roc_auc_score(yva, lr.predict_proba(Zva)[:, 1]))
            a["Gradient boosting"].append(roc_auc_score(yva, gb.predict_proba(Xva)[:, 1]))
        for k in lc:
            lc[k].append((n, np.mean(a[k]), np.std(a[k])))
    f, ax = P.fig(2.7)
    for (k, v), col in zip(lc.items(), (P.BLUE, P.ORANGE, P.AQUA)):
        v = np.array(v)
        ax.errorbar(v[:, 0], v[:, 1], yerr=v[:, 2], fmt="o-", ms=3, color=col, label=k, capsize=2)
    ax.set_xscale("log"); ax.set_xlabel("training clients"); ax.set_ylabel("validation AUC"); ax.legend(fontsize=9)
    P.save(f, "lab3_mlp_learning_curves")
    num.update({"lc_mlp_250": lc["MLP"][0][1], "lc_lr_250": lc["Logistic regression"][0][1],
                "lc_gb_250": lc["Gradient boosting"][0][1], "lc_mlp_full": lc["MLP"][-1][1],
                "lc_lr_full": lc["Logistic regression"][-1][1], "lc_gb_full": lc["Gradient boosting"][-1][1]})
    return pv_ens, pt_ens


def part_b(num, pv_mlp, pt_mlp):
    Xtr, Xva, Xte, ytr, yva, yte, Ztr, Zva, Zte = split()
    out = {}
    lr = LogisticRegression(C=10, max_iter=3000).fit(Ztr, ytr)
    out["Logistic regression (Lab 3)"] = (lr.predict_proba(Zva)[:, 1], lr.predict_proba(Zte)[:, 1], True)
    out["MLP, 2 x 64, 10-seed ensemble"] = (pv_mlp, pt_mlp, True)
    # linear SVM: C by stratified CV; scores calibrated on the validation block (Platt)
    skf = StratifiedKFold(5, shuffle=True, random_state=C.SEED)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cv = {c: np.mean([roc_auc_score(ytr[v], LinearSVC(C=c, dual=False).fit(Ztr[t], ytr[t]).decision_function(Ztr[v]))
                          for t, v in skf.split(Ztr, ytr)]) for c in (1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1)}
    c_lin = max(cv, key=cv.get)
    lsvm = LinearSVC(C=c_lin, dual=False).fit(Ztr, ytr)
    cal = CalibratedClassifierCV(FrozenEstimator(lsvm), method="sigmoid").fit(Zva, yva)
    out["Linear SVM (Platt-calibrated)"] = (cal.predict_proba(Zva)[:, 1], cal.predict_proba(Zte)[:, 1], True)
    # RBF SVM: (C, gamma) by 3-fold CV on a 6 000-client subsample, final fit on all training clients
    rng = np.random.default_rng(C.SEED); sub = rng.choice(len(Ztr), 6000, replace=False)
    sk3 = StratifiedKFold(3, shuffle=True, random_state=C.SEED)
    grid = [(c, g) for c in (0.01, 0.03, 0.1, 0.3, 1, 3) for g in (0.001, 0.003, 0.01, 0.03)]
    cvr = {cg: np.mean([roc_auc_score(ytr[sub][v], SVC(C=cg[0], gamma=cg[1]).fit(Ztr[sub][t], ytr[sub][t]).decision_function(Ztr[sub][v]))
                        for t, v in sk3.split(Ztr[sub], ytr[sub])]) for cg in grid}
    c_rbf, g_rbf = max(cvr, key=cvr.get)
    rsvm = SVC(C=c_rbf, gamma=g_rbf).fit(Ztr, ytr)
    calr = CalibratedClassifierCV(FrozenEstimator(rsvm), method="sigmoid").fit(Zva, yva)
    out["RBF SVM (Platt-calibrated)"] = (calr.predict_proba(Zva)[:, 1], calr.predict_proba(Zte)[:, 1], True)
    num.update({"svm_lin_c": c_lin, "svm_rbf_c": c_rbf, "svm_rbf_gamma": g_rbf, "svm_n_sv": int(rsvm.n_support_.sum()),
                "svm_sv_share": 100 * rsvm.n_support_.sum() / len(ytr)})
    # kNN: k by CV
    cvk = {k: np.mean([roc_auc_score(ytr[v], KNeighborsClassifier(k).fit(Ztr[t], ytr[t]).predict_proba(Ztr[v])[:, 1])
                       for t, v in skf.split(Ztr, ytr)]) for k in (5, 15, 50, 100, 200, 400, 800)}
    k_best = max(cvk, key=cvk.get)
    knn = KNeighborsClassifier(k_best).fit(Ztr, ytr)
    out[f"$k$ nearest neighbours ($k={k_best}$)"] = (knn.predict_proba(Zva)[:, 1], knn.predict_proba(Zte)[:, 1], True)
    lda = LinearDiscriminantAnalysis().fit(Ztr, ytr)
    out["Linear discriminant analysis"] = (lda.predict_proba(Zva)[:, 1], lda.predict_proba(Zte)[:, 1], True)
    nb = GaussianNB().fit(Ztr, ytr)
    out["Gaussian naive Bayes"] = (nb.predict_proba(Zva)[:, 1], nb.predict_proba(Zte)[:, 1], True)
    rows, curves = [], {}
    for name, (pv, pt, prob) in out.items():
        thr, _, _ = best_threshold(yva, pv, C_FN, C_FP)
        m = metrics(yte, pt, thr, prob)
        rows.append((name, m["AUC"], m["PR AUC"], m["Brier"], m["Cost"]))
        curves[name] = calibration_curve(yte, pt, n_bins=10, strategy="quantile")
    tab = pd.DataFrame(rows, columns=["Model", "AUC", "PR AUC", "Brier", "Cost*"]).set_index("Model")
    save_table("lab3_ext_results", tab, nd=3, col_format="p{6.3cm}cccc")
    num.update({"knn_k": k_best, "ext_auc_mlp": tab.iloc[1]["AUC"], "ext_auc_lr": tab.iloc[0]["AUC"],
                "ext_auc_rbf": tab.iloc[3]["AUC"], "ext_auc_nb": tab.iloc[6]["AUC"], "ext_brier_nb": tab.iloc[6]["Brier"],
                "ext_brier_lr": tab.iloc[0]["Brier"], "ext_brier_mlp": tab.iloc[1]["Brier"],
                "ext_cost_mlp": tab.iloc[1]["Cost*"], "ext_cost_lr": tab.iloc[0]["Cost*"],
                "ext_best_auc_name": tab["AUC"].idxmax().replace("$", "")})
    f, ax = P.fig(3.0)
    ax.plot([0, 1], [0, 1], color=P.MUTED, ls=":", lw=0.9)
    for (name, (pm, fm)), col in zip(curves.items(), P.CAT):
        ax.plot(fm, pm, "o-", ms=2.5, lw=1, color=col, label=name.replace("$", ""))
    ax.set_xlabel("mean predicted probability"); ax.set_ylabel("observed default rate"); ax.legend(fontsize=9, loc="upper left")
    P.save(f, "lab3_ext_calibration")


def part_c(num):
    g = load_german_credit()
    y = g.pop("bad").to_numpy()
    cat = [c for c in g.columns if not pd.api.types.is_numeric_dtype(g[c])]
    numc = [c for c in g.columns if c not in cat]
    enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(g[cat])
    X = np.hstack([g[numc].to_numpy(float), enc.transform(g[cat])])
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=C.SEED)
    models = {
        "Logistic regression": lambda: Pipeline([("s", StandardScaler()), ("m", LogisticRegression(C=0.3, max_iter=3000))]),
        "MLP (1 x 16, early stopping)": None,
        "RBF SVM (Platt)": lambda: Pipeline([("s", StandardScaler()), ("m", SVC(C=1, gamma="scale", probability=True, random_state=0))]),
        "$k$ nearest neighbours ($k=25$)": lambda: Pipeline([("s", StandardScaler()), ("m", KNeighborsClassifier(25))]),
        "Linear discriminant analysis": lambda: Pipeline([("s", StandardScaler()), ("m", LinearDiscriminantAnalysis())]),
        "Gaussian naive Bayes": lambda: Pipeline([("s", StandardScaler()), ("m", GaussianNB())]),
    }
    res = {k: {"auc": [], "cost": [], "brier": []} for k in models}
    thr = GERMAN_COST["fp"] / (GERMAN_COST["fp"] + GERMAN_COST["fn"])          # Bayes threshold 1/6
    for i, (t, v) in enumerate(rskf.split(X, y)):
        for name, mk in models.items():
            if mk is None:
                sc = StandardScaler().fit(X[t])
                tt, vv = train_test_split(t, test_size=0.2, stratify=y[t], random_state=i)
                m, _ = fit_mlp(sc.transform(X[tt]), y[tt], sc.transform(X[vv]), y[vv], width=16, depth=1, dropout=0.2,
                               weight_decay=1e-3, lr=1e-3, batch=64, patience=20, seed=i)
                p = predict_proba(m, sc.transform(X[v]))
            else:
                p = mk().fit(X[t], y[t]).predict_proba(X[v])[:, 1]
            res[name]["auc"].append(roc_auc_score(y[v], p))
            res[name]["brier"].append(brier_score_loss(y[v], p))
            res[name]["cost"].append(expected_cost(y[v], p, thr, GERMAN_COST["fn"], GERMAN_COST["fp"]))
    rows = [(k, np.mean(r["auc"]), np.std(r["auc"]), np.mean(r["brier"]), np.mean(r["cost"]), np.std(r["cost"]))
            for k, r in res.items()]
    base_cost = min(GERMAN_COST["fn"] * y.mean(), GERMAN_COST["fp"] * (1 - y.mean()))
    rows.append(("Trivial: reject everybody / accept everybody", 0.5, 0.0, np.nan, base_cost, 0.0))
    tab = pd.DataFrame(rows, columns=["Model", "AUC", "SD", "Brier", "Cost at 1/6", "SD "]).set_index("Model")
    save_table("lab3_german", tab, nd={"AUC": 3, "SD": 3, "Brier": 3, "Cost at 1/6": 3, "SD ": 3}, col_format="p{6.4cm}ccccc")
    num.update({"de_n": len(y), "de_bad": 100 * y.mean(), "de_p": X.shape[1], "de_thr": thr, "de_trivial_cost": base_cost,
                "de_auc_lr": tab.iloc[0]["AUC"], "de_auc_mlp": tab.iloc[1]["AUC"], "de_auc_svm": tab.iloc[2]["AUC"],
                "de_auc_sd_lr": tab.iloc[0]["SD"], "de_cost_lr": tab.iloc[0]["Cost at 1/6"],
                "de_cost_mlp": tab.iloc[1]["Cost at 1/6"], "de_best_name": tab["AUC"].iloc[:-1].idxmax().replace("$", "")})


def main():
    num = {}
    pv, pt = part_a(num)
    part_b(num, pv, pt)
    part_c(num)
    save_numbers("lab3x", num)
    print(num)


if __name__ == "__main__":
    main()
