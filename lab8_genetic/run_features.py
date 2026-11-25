"""Lab 8.2. Genetic algorithms for the Lab 7 task.

(a) feature selection: a 20-bit chromosome switches individual features on
    or off (Lab 7 worked with 6 groups); fitness = pooled RMSE of the
    rolling ridge forecasts over the training+validation decision points
    plus a parsimony penalty;
(b) learning process: genes are the hyperparameters of the Q-learning
    agent and of its reward (alpha, gamma, exploration schedule, feature
    and switching costs, number of episodes); fitness = validation RMSE.

Run:  python -m lab8_genetic.run_features
"""
from __future__ import annotations

import json
import pickle
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from dmfin import config as C
from dmfin import plotting as P
from dmfin.reporting import save_numbers, save_table
from lab7_rl_features.env import H, STEP, WINDOW, FeatureSelectionEnv, QAgent, pooled, purged_blocks, run_episode
from lab7_rl_features.run import get_tables
from lab7_rl_features.stream import FEATURES

from .ga import GeneticAlgorithm

ARRAYS = C.RESULTS_DIR / "lab8_step_arrays.pkl"
PARSIMONY = 0.001


def step_arrays(store, T):
    """Per decision point: standardised training/evaluation matrices of all 20 features."""
    if ARRAYS.exists():
        return pickle.loads(ARRAYS.read_bytes())
    out = []
    dates = T.dates
    for s in dates:
        w = store.window((s - pd.Timedelta(days=1200)).strftime("%Y-%m-%d"),
                         (s + pd.Timedelta(days=45)).strftime("%Y-%m-%d"))
        X = w.drop(columns="y").ffill(limit=5)
        past = X.loc[:s]; lab = w["y"].loc[:s].iloc[:-H]
        trn = lab.index[-WINDOW:]
        ev = X.index[X.index > s][:STEP]; ev = ev[w.loc[ev, "y"].notna()]
        Xt = past.loc[trn, FEATURES]
        ok = (Xt.notna().mean() >= 0.8).to_numpy() & X.loc[ev, FEATURES].notna().all().to_numpy()
        Xt = Xt.fillna(Xt.mean()).fillna(0).to_numpy()
        Xe = X.loc[ev, FEATURES].fillna(0).to_numpy()
        mu, sd = Xt.mean(0), Xt.std(0) + 1e-9
        out.append(((Xt - mu) / sd, lab.loc[trn].to_numpy(), (Xe - mu) / sd, w.loc[ev, "y"].to_numpy(), ok))
    ARRAYS.write_bytes(pickle.dumps(out))
    return out


def ridge_rmse(arr, mask, alpha=1.0):
    Zt, yt, Ze, ye, ok = arr
    m = np.asarray(mask, bool) & ok
    if not m.any():
        return float(np.sqrt(np.mean((ye - yt.mean()) ** 2)))
    A, B = Zt[:, m], Ze[:, m]
    ym = yt.mean()
    w = np.linalg.solve(A.T @ A + alpha * np.eye(m.sum()), A.T @ (yt - ym))
    return float(np.sqrt(np.mean((B @ w + ym - ye) ** 2)))


def score(arrs, nobs, mask):
    return pooled(np.array([ridge_rmse(a, mask) for a in arrs]), nobs)


# ------------------------------------------------------- (b) agent genes
AG = {"alpha": [0.02, 0.05, 0.1, 0.2, 0.3], "gamma": [0.0, 0.3, 0.5, 0.7, 0.9, 0.95],
      "eps0": [0.2, 0.5, 0.8], "eps_min": [0.01, 0.05, 0.1], "lam": [0.0, 0.02, 0.05, 0.1],
      "psi": [0.0, 0.05, 0.1, 0.2], "episodes": [100, 200, 300]}
AK = list(AG)


def agent_val(g, seeds=(0, 1, 2)):
    _, T = get_tables()
    cfg = {k: AG[k][g[i]] for i, k in enumerate(AK)}
    tr, va, _ = purged_blocks(T.dates, T.ends)
    vals = []
    for seed in seeds:
        env = FeatureSelectionEnv(T, tr); env.LAMBDA, env.PSI = cfg["lam"], cfg["psi"]
        ag = QAgent(alpha=cfg["alpha"], gamma=cfg["gamma"], seed=seed)
        for ep in range(cfg["episodes"]):
            ag.eps = max(cfg["eps_min"], cfg["eps0"] * (1 - ep / (0.8 * cfg["episodes"])))
            run_episode(env, ag)
        ag.eps = cfg["eps_min"]
        ev = FeatureSelectionEnv(T, va); ev.LAMBDA, ev.PSI = cfg["lam"], cfg["psi"]
        e, _, _ = run_episode(ev, ag)
        vals.append(pooled(e, T.nobs[va]))
    return float(np.mean(vals))


def agent_test(g, seeds=range(10)):
    _, T = get_tables()
    cfg = {k: AG[k][g[i]] for i, k in enumerate(AK)}
    tr, va, te = purged_blocks(T.dates, T.ends)
    out = []
    for seed in seeds:
        env = FeatureSelectionEnv(T, tr); env.LAMBDA, env.PSI = cfg["lam"], cfg["psi"]
        ag = QAgent(alpha=cfg["alpha"], gamma=cfg["gamma"], seed=seed)
        for ep in range(cfg["episodes"]):
            ag.eps = max(cfg["eps_min"], cfg["eps0"] * (1 - ep / (0.8 * cfg["episodes"])))
            run_episode(env, ag)
        ag.eps = cfg["eps_min"]
        for steps in (va, te):
            ev = FeatureSelectionEnv(T, steps); ev.LAMBDA, ev.PSI = cfg["lam"], cfg["psi"]
            e, _, _ = run_episode(ev, ag)
        out.append(pooled(e, T.nobs[te]))
    return float(np.mean(out)), float(np.std(out))


def main():
    store, T = get_tables()
    arrs = step_arrays(store, T)
    tr, va, te = purged_blocks(T.dates, T.ends)
    trva = np.r_[tr, va]
    A_trva, A_te = [arrs[k] for k in trva], [arrs[k] for k in te]
    n_trva, n_te = T.nobs[trva], T.nobs[te]
    fit = lambda g: score(A_trva, n_trva, g) + PARSIMONY * sum(g)
    num = {"parsimony": PARSIMONY}

    # ----------------------------------------------- (a) GA over 20 feature bits
    runs = []
    for seed in range(5):
        ga = GeneticAlgorithm([2] * len(FEATURES), fitness=fit, pop_size=40, generations=40, tournament=3,
                              p_cx=0.9, p_mut=1 / len(FEATURES), elite=2, seed=seed,
                              repair=lambda g: g if any(g) else (1,) + g[1:])
        runs.append(ga.run())
    best_run = min(runs, key=lambda r: r.best_fit)
    best = best_run.best
    sel = [f for f, b in zip(FEATURES, best) if b]
    all_mask = [1] * len(FEATURES)
    group_mask = [1 if f in ("log_pk_d", "log_pk_w", "log_pk_m", "log_vix", "vix_chg_5", "vrp") else 0 for f in FEATURES]
    rows = [("All 20 features", 20, score(A_trva, n_trva, all_mask), score(A_te, n_te, all_mask)),
            ("Groups vol + vix (best fixed subset, Lab 7)", 6, score(A_trva, n_trva, group_mask), score(A_te, n_te, group_mask)),
            ("GA, individual features (best of 5 runs)", len(sel), score(A_trva, n_trva, best), score(A_te, n_te, best))]
    # forward stepwise selection as the classical wrapper baseline
    cur, path = [0] * len(FEATURES), []
    for _ in range(len(FEATURES)):
        cand = [(fit(tuple(cur[:i] + [1] + cur[i + 1:])), i) for i in range(len(FEATURES)) if not cur[i]]
        v, i = min(cand)
        if path and v >= path[-1][0]:
            break
        cur[i] = 1; path.append((v, i))
    rows.append(("Forward stepwise selection (same fitness)", sum(cur), score(A_trva, n_trva, cur), score(A_te, n_te, cur)))
    tab = pd.DataFrame(rows, columns=["Feature set", "Features", "Train+valid. RMSE", "Test RMSE"]).set_index("Feature set")
    save_table("lab8_features", tab, nd={"Features": 0, "Train+valid. RMSE": 4, "Test RMSE": 4}, col_format="p{7.4cm}ccc")
    print(tab.to_string())
    num.update({"ga_n_sel": len(sel), "ga_sel": ", ".join(sel).replace("_", "\\_"),
                "ga_test": tab.iloc[2, 2], "all_test": tab.iloc[0, 2], "grp_test": tab.iloc[1, 2],
                "fwd_test": tab.iloc[3, 2], "fwd_sel": ", ".join(f for f, b in zip(FEATURES, cur) if b).replace("_", "\\_"),
                "ga_evals": best_run.evaluations, "ga_runs_agree": float(np.mean([r.best == best for r in runs])),
                "freq_sel_min": float(np.min(np.mean([r.best for r in runs], axis=0)[np.array(best, bool)]))})
    freq = np.mean([r.best for r in runs], axis=0)
    f, axs = P.fig(2.8, ncols=2, gridspec_kw={"width_ratios": [1, 1.25]})
    for r_, col in zip(runs, P.CAT):
        h = np.array(r_.history)
        axs[0].plot(h[:, 0], h[:, 1], color=col, lw=1.0)
    axs[0].set_xlabel("generation"); axs[0].set_ylabel("best fitness (RMSE + penalty)")
    axs[1].barh(range(len(FEATURES)), freq, color=P.BLUE)
    axs[1].set_yticks(range(len(FEATURES))); axs[1].set_yticklabels(FEATURES, fontsize=7)
    axs[1].invert_yaxis(); axs[1].set_xlabel("share of GA runs selecting the feature")
    f.tight_layout(); P.save(f, "lab8_features")

    # ------------------------------------- (b) GA over the agent's learning process
    # the setting of Lab 7: alpha and gamma chosen there on the validation points, the fixed exploration schedule
    # (0.5 decreasing to 0.05), the costs of the reward and the number of episodes
    l7 = json.loads((C.RESULTS_DIR / "lab7_numbers.json").read_text())
    lab7_cfg = {"alpha": l7["alpha"], "gamma": l7["gamma"], "eps0": 0.5, "eps_min": 0.05, "lam": l7["lambda"],
                "psi": l7["psi"], "episodes": l7["episodes"]}
    lab7_g = tuple(AG[k].index(lab7_cfg[k]) for k in AK)
    pool = ProcessPoolExecutor(12)
    ga = GeneticAlgorithm([len(v) for v in AG.values()], batch_fitness=lambda gs: list(pool.map(agent_val, gs)),
                          pop_size=20, generations=8, tournament=3, p_cx=0.9, p_mut=1 / len(AK), elite=2, seed=C.SEED,
                          init=[lab7_g])
    res = ga.run()
    (t_ga, s_ga), (t_lab7, s_lab7) = pool.map(agent_test, [res.best, lab7_g])
    pool.shutdown()
    cfg = {k: AG[k][res.best[i]] for i, k in enumerate(AK)}
    gt = pd.DataFrame([{"Setting": "Lab 7 (grid search)", **{k: AG[k][lab7_g[i]] for i, k in enumerate(AK)},
                        "Valid. RMSE": res.cache[lab7_g], "Test RMSE": t_lab7},
                       {"Setting": "GA", **cfg, "Valid. RMSE": res.best_fit, "Test RMSE": t_ga}]).set_index("Setting")
    gt.columns = ["$\\alpha$", "$\\gamma$", "$\\varepsilon_0$", "$\\varepsilon_{\\min}$", "$\\lambda$", "$\\psi$",
                  "Episodes", "Valid. RMSE", "Test RMSE"]
    save_table("lab8_agent", gt, nd={"$\\alpha$": 2, "$\\gamma$": 2, "$\\varepsilon_0$": 1, "$\\varepsilon_{\\min}$": 2,
                                     "$\\lambda$": 2, "$\\psi$": 2, "Episodes": 0, "Valid. RMSE": 4, "Test RMSE": 4})
    print(gt.to_string())
    num.update({"agent_ga_val": res.best_fit, "agent_lab7_val": res.cache[lab7_g], "agent_ga_test": t_ga,
                "agent_lab7_test": t_lab7, "agent_ga_sd": s_ga, "agent_lab7_sd": s_lab7, "agent_evals": res.evaluations,
                **{f"agent_{k}": cfg[k] for k in AK}})
    save_numbers("lab8f", num)
    print(num)


if __name__ == "__main__":
    main()
