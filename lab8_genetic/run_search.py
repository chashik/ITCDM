"""Lab 8 (iteration 2), part A. Search methods at an equal budget, nested validation, the winner's curse.

Every method (random search with one or two seeds per configuration, GA, TPE, Hyperband,
PBT) tunes the global MLP of ``search.py`` five times (five search seeds) with a budget of
600 training epochs.  The winner of every search is then retrained with three new seeds:
the difference between its validation score in the search and after retraining measures
the optimism of selecting the best of many noisy scores (the winner's curse); validation
block B, never used by the search, gives a nested check.

Run:  python -m lab8_genetic.run_search
"""
from __future__ import annotations

import multiprocessing as mp
import pickle
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression

from dmfin import config as C
from dmfin import plotting as P
from dmfin.reporting import save_numbers, save_table

from . import search as S

SEEDS = range(5)
N_FRESH = 3
N_DEFAULT = 10
CACHE = C.RESULTS_DIR / "lab8x_cache"
NAMES = {"random": "Random search", "random2": "Random search, 2 seeds per configuration", "ga": "Genetic algorithm",
         "tpe": "TPE (Optuna)", "hyperband": "Hyperband", "pbt": "Population-based training"}
HAR = ["lrv_d", "lrv_w", "lrv_m", "log_vix"]
_D = {}


def data() -> S.Data:
    if "d" not in _D:
        import torch
        torch.set_num_threads(1)
        _D["d"] = S.Data()
    return _D["d"]


def cached(name, fn):
    f = CACHE / f"{name}.pkl"
    if f.exists():
        return pickle.loads(f.read_bytes())
    t0 = time.time()
    out = fn()
    out["minutes"] = (time.time() - t0) / 60
    CACHE.mkdir(parents=True, exist_ok=True)
    f.write_bytes(pickle.dumps(out))
    print(f"{time.strftime('%H:%M:%S')} {name} done ({out['minutes']:.1f} min)", flush=True)
    return out


def _search(args):
    method, seed = args
    return cached(f"{method}_{seed}", lambda: S.METHODS[method](data(), seed))


def _metrics(r: S.Run) -> dict:
    return {"valA": r.mse("valA"), "valB": r.mse("valB"), "test": r.mse("test"), "pred_test": r.predict("test"),
            "epochs": r.epochs}


def _fresh(args):
    method, seed, j = args
    res = _search((method, seed))
    return cached(f"fresh_{method}_{seed}_{j}", lambda: _metrics(S.replay(data(), res, 10_000 + 100 * seed + j)))


def _default(j):
    def fn():
        r = S.Run(S.DEFAULT, 20_000 + j, data())
        r.fit(S.MAX_EPOCHS, S.PATIENCE)
        return _metrics(r)
    return cached(f"default_{j}", fn)


def har_baseline(d: S.Data) -> dict:
    """A linear HAR + VIX model for each series, fitted on the training block."""
    p = d.panel
    out = {b: np.full(len(d.rows[b]), np.nan) for b in ("valA", "valB", "test")}
    ser = p["series"].to_numpy()
    for s in np.unique(ser):
        tr = d.rows["train"][ser[d.rows["train"]] == s]
        if len(tr) < 200:
            continue
        m = LinearRegression().fit(p.loc[tr, HAR], p.loc[tr, "y_5"])
        for b in out:
            sel = ser[d.rows[b]] == s
            if sel.any():
                out[b][sel] = m.predict(p.loc[d.rows[b][sel], HAR])
    return out


def main():
    t0 = time.time()
    ctx = mp.get_context("spawn")
    jobs = [(m, s) for m in S.METHODS for s in SEEDS]
    with ProcessPoolExecutor(12, mp_context=ctx) as ex:
        res = dict(zip(jobs, ex.map(_search, jobs)))
        fresh_jobs = [(m, s, j) for m, s in jobs for j in range(N_FRESH)]
        fr = dict(zip(fresh_jobs, ex.map(_fresh, fresh_jobs)))
        dfl = list(ex.map(_default, range(N_DEFAULT)))
    d = data()
    rt = lambda x: float(np.sqrt(x))
    yb = {b: d.y[d.rows[b]].numpy() for b in ("valA", "valB", "test")}
    har = har_baseline(d)
    ok = np.isfinite(har["test"])
    har_rmse = {b: rt(np.nanmean((har[b] - yb[b]) ** 2)) for b in har}
    num = {"budget": S.BUDGET, "max_epochs": S.MAX_EPOCHS, "patience": S.PATIENCE, "n_seeds": len(SEEDS),
           "n_fresh": N_FRESH, "space_size": f"{int(np.prod(S.CARD)):,}".replace(",", "\\,"),
           "n_train": len(d.rows["train"]), "n_valA": len(d.rows["valA"]), "n_valB": len(d.rows["valB"]),
           "n_test": len(d.rows["test"]), "har_test": har_rmse["test"], "har_valA": har_rmse["valA"],
           "har_valB": har_rmse["valB"], "har_share_ok": 100 * ok.mean()}

    # ---------------- seed noise of one configuration
    dv = np.array([rt(x["valA"]) for x in dfl]); dt = np.array([rt(x["test"]) for x in dfl])
    num.update({"def_valA": dv.mean(), "def_valA_sd": dv.std(ddof=1), "def_test": dt.mean(), "def_test_sd": dt.std(ddof=1),
                "def_epochs": float(np.mean([x["epochs"] for x in dfl]))})

    # ---------------- table of methods
    rows, wins = [], []
    for m in S.METHODS:
        r = [res[(m, s)] for s in SEEDS]
        fa = [np.mean([rt(fr[(m, s, j)]["valA"]) for j in range(N_FRESH)]) for s in SEEDS]
        fb = [np.mean([rt(fr[(m, s, j)]["valB"]) for j in range(N_FRESH)]) for s in SEEDS]
        ft = [np.mean([rt(fr[(m, s, j)]["test"]) for j in range(N_FRESH)]) for s in SEEDS]
        sa = [rt(x["valA"]) for x in r]
        n_cfg = [len(x["evals"]) for x in r]
        opt = np.array(fa) - np.array(sa)
        rows.append((NAMES[m], np.mean(n_cfg), np.mean(sa), np.mean(fa), np.mean(opt), np.mean(fb), np.mean(ft), np.std(ft),
                     np.mean([x["minutes"] for x in r])))
        for s, a, b, c in zip(SEEDS, sa, fa, ft):
            wins.append((m, s, a, b, c))
        num.update({f"{m}_cfgs": float(np.mean(n_cfg)), f"{m}_search": float(np.mean(sa)), f"{m}_fresh": float(np.mean(fa)),
                    f"{m}_opt": float(np.mean(opt)), f"{m}_valB": float(np.mean(fb)), f"{m}_test": float(np.mean(ft)),
                    f"{m}_test_sd": float(np.std(ft)), f"{m}_minutes": float(np.mean([x["minutes"] for x in r]))})
    tab = pd.DataFrame(rows, columns=["Method", "Configurations", "Valid. A (search)", "Valid. A (retrained)", "Optimism",
                                      "Valid. B", "Test", "Test SD", "Minutes"]).set_index("Method")
    save_table("lab8x_search", tab, nd={"Configurations": 0, "Valid. A (search)": 4, "Valid. A (retrained)": 4,
                                        "Optimism": 4, "Valid. B": 4, "Test": 4, "Test SD": 4, "Minutes": 1},
               col_format="p{4.2cm}cccccccc")
    print(tab.round(4).to_string())
    best_m = tab["Valid. B"].idxmin()
    num.update({"best_valB_method": best_m, "best_test_method": tab["Test"].idxmin(),
                "test_range": float(tab["Test"].max() - tab["Test"].min())})
    print(f"wall time of this run (searches read from the cache are not counted): {(time.time() - t0) / 60:.1f} min")
    # winners' configurations
    wrows = []
    for m in S.METHODS:
        for s in SEEDS:
            w = res[(m, s)]["winner"]
            wrows.append((NAMES[m], s, w["lookback"], w["width"], w["layers"], w["dropout"], w["lr"], w["wd"], w["batch"]))
    wdf = pd.DataFrame(wrows, columns=["Method", "Seed", "L", "Width", "Layers", "Dropout", "LR", "WD", "Batch"])
    num.update({"win_lookback_mode": int(wdf["L"].mode()[0]), "win_layers_mode": int(wdf["Layers"].mode()[0]),
                "win_width_mode": int(wdf["Width"].mode()[0])})
    summ = wdf.groupby("Method", sort=False).agg(L=("L", lambda v: "/".join(str(x) for x in v)),
                                                 Width=("Width", lambda v: "/".join(str(x) for x in v)),
                                                 Layers=("Layers", lambda v: "/".join(str(x) for x in v)))
    save_table("lab8x_winners", summ, nd=0, col_format="p{5.2cm}ccc")

    # ---------------- nested validation: ranks on validation A and B of all evaluated configurations
    ev = [(m, e) for (m, s), r in res.items() if m in ("random", "tpe", "ga") for e in r["evals"]]
    a = np.array([rt(e[1]) for _, e in ev]); b = np.array([rt(e[2]) for _, e in ev])
    rho = spearmanr(a, b).statistic
    top = np.argsort(a)[:20]
    rho_top = spearmanr(a[top], b[top]).statistic
    num.update({"n_evals_rank": len(a), "rho_ab": float(rho), "rho_ab_top20": float(rho_top)})
    best_cfg = np.argmin(a)
    num["rank_on_B_of_best_on_A"] = int((b < b[best_cfg]).sum() + 1)

    # ---------------- ensemble of the 30 retrained winners vs HAR
    pt = np.mean([fr[k]["pred_test"] for k in fr], axis=0)
    num["ens_test"] = rt(np.mean((pt - yb["test"]) ** 2))
    num["ens_ratio_har"] = num["ens_test"] / har_rmse["test"]

    # ---------------- figures
    f, axs = P.fig(2.9, ncols=2)
    cols = dict(zip(S.METHODS, [P.ORANGE, P.MUTED, P.BLUE, P.AQUA, P.RED, P.CAT[5] if len(P.CAT) > 5 else P.INK2]))
    grid = np.arange(0, S.BUDGET + 1, 10)
    for m in S.METHODS:
        curves = []
        for s in SEEDS:
            e = res[(m, s)]["evals"]
            used = np.array([x[3] for x in e]); sc = np.minimum.accumulate([rt(x[1]) for x in e])
            curves.append([sc[used <= g].min() if (used <= g).any() else np.nan for g in grid])
        cv = np.nanmean(np.array(curves, float), axis=0)
        lab = NAMES[m].replace(", 2 seeds per configuration", " (2 seeds)")
        if np.isfinite(cv).sum() <= 1:          # PBT: the population is evaluated only at the end of the budget
            axs[0].plot(grid[np.isfinite(cv)], cv[np.isfinite(cv)], "o", ms=4, color=cols[m], label=lab + " (end)")
        else:
            axs[0].plot(grid, cv, color=cols[m], lw=1.1, label=lab)
    axs[0].axhline(num["def_valA"], color=P.INK2, ls=":", lw=0.9, label="default configuration")
    axs[0].set_xlabel("training epochs used"); axs[0].set_ylabel("best validation-A RMSE so far"); axs[0].legend(fontsize=6.5)
    lo = min(rt(res[k]["valA"]) for k in res)
    axs[0].set_ylim(lo - 0.003, num["def_valA"] + 0.03)
    for m in S.METHODS:
        pts = [(w[2], w[3]) for w in wins if w[0] == m]
        axs[1].scatter(*zip(*pts), s=14, color=cols[m])
    lim = [min(w[2] for w in wins) - 0.002, max(w[3] for w in wins) + 0.002]
    axs[1].plot(lim, lim, color=P.INK2, lw=0.7)
    axs[1].set_xlabel("validation-A RMSE of the winner in the search"); axs[1].set_ylabel("after retraining (3 new seeds)")
    f.tight_layout(); P.save(f, "lab8x_search")

    f, axs = P.fig(2.6, ncols=2)
    axs[0].scatter(a, b, s=4, color=P.BLUE, alpha=0.5)
    axs[0].scatter(a[top], b[top], s=8, color=P.ORANGE)
    axs[0].set_xlabel("validation A (2015--2016) RMSE"); axs[0].set_ylabel("validation B (2017--2018) RMSE")
    axs[0].set_xlim(a.min() - 0.005, np.quantile(a, 0.9)); axs[0].set_ylim(b.min() - 0.005, np.quantile(b, 0.9))
    axs[1].hist(dv, bins=8, color=P.MUTED, alpha=0.8, label="default, 10 seeds")
    axs[1].hist([w[3] for w in wins], bins=12, color=P.BLUE, alpha=0.6, label="30 winners, retrained")
    axs[1].set_xlabel("validation-A RMSE"); axs[1].legend(fontsize=7)
    f.tight_layout(); P.save(f, "lab8x_nested")
    save_numbers("lab8x", num)
    print(num)


if __name__ == "__main__":
    main()
