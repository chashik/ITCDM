"""Lab 8 (iteration 2), part B. Genetic feature selection on the 50-feature stream of Lab 7.

The chromosome has one bit per feature (2^50 subsets).  The fitness is the pooled RMSE of
the rolling ridge forecasts of Lab 7 over a block of decision points plus a parsimony
penalty.  Two designs are compared:

* search on training + validation (2003-2018), as in Lab 8.2, and report the test error;
* nested: search on training (2003-2014) only, check the selected subset on validation
  (2015-2018) and report the test error.

Controls with the same number of fitness evaluations: random subsets; and forward stepwise
selection.  The difference between the fitness of the winner and its error on new data
measures the optimism of the search (the winner's curse).

Run:  python -m lab8_genetic.run_features_ext
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from dmfin import config as C
from dmfin import plotting as P
from dmfin.reporting import save_numbers, save_table
from lab7_rl_features import ext as X
from lab7_rl_features.env import purged_blocks
from lab7_rl_features.run_ext import get_frame, get_tables

from .ga import GeneticAlgorithm

PARSIMONY = 0.001
FEATS = X.features(X.GROUPS11)
POP, GENS, RUNS = 40, 40, 5
_A = {}


def step_arrays(frame: pd.DataFrame, dates: pd.DatetimeIndex) -> list:
    """Per decision point: standardized training and evaluation matrices of the 50 features."""
    out = []
    for s in dates:
        p = frame.index.get_loc(s)
        w = frame.iloc[max(0, p - X.WINDOW - 30):min(len(frame), p + X.STEP + 1)]
        Xw = w[FEATS].ffill(limit=5)
        past = Xw.loc[:s]
        lab = w["y"].loc[:s].iloc[:-X.H]
        trn = lab.index[-X.WINDOW:]
        ev = Xw.index[Xw.index > s][:X.STEP]
        ev = ev[w.loc[ev, "y"].notna()]
        Xt = past.loc[trn, FEATS]
        ok = (Xt.notna().mean() >= 0.8).to_numpy() & Xw.loc[ev, FEATS].notna().all().to_numpy()
        Xt = Xt.fillna(Xt.mean()).fillna(0).to_numpy()
        Xe = Xw.loc[ev, FEATS].fillna(0).to_numpy()
        mu, sd = Xt.mean(0), Xt.std(0) + 1e-9
        out.append((((Xt - mu) / sd).astype(np.float32), lab.loc[trn].to_numpy(), ((Xe - mu) / sd).astype(np.float32),
                    w.loc[ev, "y"].to_numpy(), ok))
    return out


def ridge_rmse(arr, mask, alpha=1.0):
    Zt, yt, Ze, ye, ok = arr
    m = np.asarray(mask, bool) & ok
    if not m.any():
        return float(np.sqrt(np.mean((ye - yt.mean()) ** 2)))
    A, B = Zt[:, m].astype(float), Ze[:, m].astype(float)
    ym = yt.mean()
    w = np.linalg.solve(A.T @ A + alpha * np.eye(m.sum()), A.T @ (yt - ym))
    return float(np.sqrt(np.mean((B @ w + ym - ye) ** 2)))


def score(block, mask):
    arrs, nobs = _A[block]
    return X.pooled(np.array([ridge_rmse(a, mask) for a in arrs]), nobs)


def fitness(block, g):
    return score(block, g) + PARSIMONY * sum(g)


def _init(blocks):
    _A.update(blocks)


def _repair(g):
    return g if any(g) else (1,) + tuple(g[1:])


def _ga(args):
    block, seed = args
    ga = GeneticAlgorithm([2] * len(FEATS), fitness=lambda g: fitness(block, g), pop_size=POP, generations=GENS,
                          tournament=3, p_cx=0.9, p_mut=1 / len(FEATS), elite=2, seed=seed, repair=_repair)
    res = ga.run()
    return res.best, res.best_fit, res.evaluations, [h[1] for h in res.history]


def _random(args):
    block, seed, n = args
    rng = np.random.default_rng(seed)
    best, bf = None, np.inf
    for _ in range(n):
        q = rng.uniform()                                   # subset sizes of all orders are sampled
        g = _repair(tuple(int(b) for b in rng.uniform(size=len(FEATS)) < q))
        v = fitness(block, g)
        if v < bf:
            best, bf = g, v
    return best, bf


def _forward(block):
    cur, path = [0] * len(FEATS), []
    while True:
        cand = [(fitness(block, tuple(cur[:i] + [1] + cur[i + 1:])), i) for i in range(len(FEATS)) if not cur[i]]
        if not cand:
            break
        v, i = min(cand)
        if path and v >= path[-1][0]:
            break
        cur[i] = 1; path.append((v, i))
    return tuple(cur), path[-1][0]


def main():
    frame = get_frame()
    T = get_tables()["11"]
    arrs = step_arrays(frame, T.dates)
    tr, va, te = purged_blocks(T.dates, T.ends)
    trva = np.r_[tr, va]
    blocks = {b: ([arrs[k] for k in ix], T.nobs[ix]) for b, ix in (("tr", tr), ("va", va), ("trva", trva), ("te", te))}
    _init(blocks)
    num = {"parsimony": PARSIMONY, "n_feats": len(FEATS), "pop": POP, "gens": GENS, "runs": RUNS}

    with ProcessPoolExecutor(12, initializer=_init, initargs=(blocks,)) as ex:
        ga_jobs = [(b, s) for b in ("trva", "tr") for s in range(RUNS)]
        ga = dict(zip(ga_jobs, ex.map(_ga, ga_jobs)))
        n_eval = int(np.mean([ga[j][2] for j in ga_jobs]))
        rnd_jobs = [(b, s, n_eval) for b in ("trva", "tr") for s in range(RUNS)]
        rnd = dict(zip([(b, s) for b, s, _ in rnd_jobs], ex.map(_random, rnd_jobs)))
        fwd = dict(zip(("trva", "tr"), ex.map(_forward, ("trva", "tr"))))

    rows = []
    sc = lambda b, g: score(b, g)
    allm = (1,) * len(FEATS)
    vv = tuple(int(f in X.GROUPS11["vol"] + X.GROUPS11["vix"]) for f in FEATS)
    rows.append(("All 50 features", len(FEATS), np.nan, sc("tr", allm), sc("va", allm), sc("te", allm)))
    rows.append(("Groups vol + vix (best fixed group subset, Lab 7)", sum(vv), np.nan, sc("tr", vv), sc("va", vv), sc("te", vv)))
    for b, lab in (("trva", "search on 2003--2018"), ("tr", "search on 2003--2014 (nested)")):
        best_seed = min(range(RUNS), key=lambda s: ga[(b, s)][1])
        g = ga[(b, best_seed)][0]
        rows.append((f"GA, {lab}", sum(g), ga[(b, best_seed)][1] - PARSIMONY * sum(g), sc("tr", g), sc("va", g), sc("te", g)))
        num[f"ga_{b}_n"] = sum(g)
        num[f"ga_{b}_sel"] = ", ".join(f for f, x in zip(FEATS, g) if x).replace("_", "\\_")
        num[f"ga_{b}_test"] = sc("te", g); num[f"ga_{b}_val"] = sc("va", g); num[f"ga_{b}_fit"] = rows[-1][2]
        num[f"ga_{b}_test_runs_sd"] = float(np.std([sc("te", ga[(b, s)][0]) for s in range(RUNS)]))
        num[f"ga_{b}_agree"] = float(np.mean([ga[(b, s)][0] == g for s in range(RUNS)]))
        num[f"ga_{b}_agree_n"] = int(sum(ga[(b, s)][0] == g for s in range(RUNS)))
        rb = min(range(RUNS), key=lambda s: rnd[(b, s)][1])
        gr = rnd[(b, rb)][0]
        rows.append((f"Random subsets, same evaluations, {lab}", sum(gr), rnd[(b, rb)][1] - PARSIMONY * sum(gr), sc("tr", gr),
                     sc("va", gr), sc("te", gr)))
        num[f"rnd_{b}_test"] = sc("te", gr); num[f"rnd_{b}_n"] = sum(gr); num[f"rnd_{b}_fit"] = rows[-1][2]
        num[f"rnd_{b}_val"] = sc("va", gr)
        num[f"rnd_{b}_sel"] = ", ".join(f for f, x in zip(FEATS, gr) if x).replace("_", "\\_")
        gf = fwd[b][0]
        rows.append((f"Forward stepwise, {lab}", sum(gf), fwd[b][1] - PARSIMONY * sum(gf), sc("tr", gf), sc("va", gf), sc("te", gf)))
        num[f"fwd_{b}_test"] = sc("te", gf); num[f"fwd_{b}_n"] = sum(gf)
        num[f"fwd_{b}_sel"] = ", ".join(f for f, x in zip(FEATS, gf) if x).replace("_", "\\_")
    tab = pd.DataFrame(rows, columns=["Feature set", "Features", "Search RMSE", "Train", "Valid.", "Test"]).set_index("Feature set")
    save_table("lab8x_features", tab, nd={"Features": 0, "Search RMSE": 4, "Train": 4, "Valid.": 4, "Test": 4},
               col_format="p{6.6cm}ccccc")
    print(tab.round(4).to_string())
    num.update({"n_eval": n_eval, "all_test": rows[0][5], "vv_test": rows[1][5], "vv_val": rows[1][4], "vv_train": rows[1][3],
                "opt_trva": float(tab.iloc[2]["Test"] - tab.iloc[2]["Search RMSE"]),
                "opt_tr": float(tab.iloc[5]["Valid."] - tab.iloc[5]["Search RMSE"])})

    # ---------------- figures: convergence and selection frequency
    f, axs = P.fig(3.6, ncols=2, gridspec_kw={"width_ratios": [1, 1.2]})
    for b, col in (("trva", P.BLUE), ("tr", P.ORANGE)):
        for s in range(RUNS):
            axs[0].plot(ga[(b, s)][3], color=col, lw=0.9, alpha=0.8, label=("search on 2003--2018" if b == "trva"
                                                                             else "search on 2003--2014") if s == 0 else None)
    axs[0].set_xlabel("generation"); axs[0].set_ylabel("best fitness (RMSE + penalty)"); axs[0].legend(fontsize=7)
    freq = {b: np.mean([ga[(b, s)][0] for s in range(RUNS)], axis=0) for b in ("trva", "tr")}
    keep = [i for i in range(len(FEATS)) if freq["trva"][i] > 0 or freq["tr"][i] > 0]
    y = np.arange(len(keep))
    axs[1].barh(y - 0.2, freq["trva"][keep], height=0.38, color=P.BLUE, label="2003--2018")
    axs[1].barh(y + 0.2, freq["tr"][keep], height=0.38, color=P.ORANGE, label="2003--2014")
    axs[1].set_yticks(y); axs[1].set_yticklabels([FEATS[i] for i in keep], fontsize=7); axs[1].invert_yaxis()
    axs[1].set_xlabel("share of GA runs selecting the feature"); axs[1].legend(fontsize=7)
    f.tight_layout(); P.save(f, "lab8x_features")
    num["n_ever_selected"] = len(keep)
    save_numbers("lab8xf", num)
    print(num)


if __name__ == "__main__":
    main()
