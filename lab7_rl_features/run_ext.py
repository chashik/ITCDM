"""Lab 7 (iteration 2). A wider stream, further agents and the unlabeled-data variant.

* The documents of the Lab 7 store receive 30 new fields in five groups (``ext.new_fields``);
  the 20 old features are read back unchanged, so the 6-group results can be reproduced.
* Tables are built for the 6 old groups (63 subsets) and for all 11 groups (2047 subsets).
* Agents: tabular Q-learning and SARSA, the LinUCB contextual bandit and a deep Q-network;
  hyperparameters on the validation block (3 seeds), final runs with 10 seeds; all agents
  learn from a reward only after it is realized (one decision later).
* Unlabeled variant: the reward is the reconstruction error of all available features by
  the chosen groups, so no labels are needed; the forecast RMSE is used only for evaluation.

Run:  python -m lab7_rl_features.run_ext
"""
from __future__ import annotations

import os
import pickle
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from dmfin import config as C
from dmfin import plotting as P
from dmfin.reporting import save_numbers, save_table

from . import ext as X
from .env import purged_blocks
from .stream import MarketStore

STORE = C.RESULTS_DIR / "lab7_stream.json"
TABLES = C.RESULTS_DIR / "lab7x_tables.pkl"
RUNS = C.RESULTS_DIR / "lab7x_runs.pkl"
FRAME = C.RESULTS_DIR / "lab7x_frame.parquet"
EPISODES = {"q": 300, "sarsa": 300, "linucb": 30, "dqn": 150}
GRID = {"q": [{"alpha": a, "gamma": g} for a in (0.05, 0.1, 0.3) for g in (0.5, 0.9)],
        "sarsa": [{"alpha": a, "gamma": g} for a in (0.05, 0.1, 0.3) for g in (0.5, 0.9)],
        "linucb": [{"alpha": a} for a in (0.1, 0.3, 1.0)],
        "dqn": [{"lr": lr, "gamma": g} for lr in (3e-4, 1e-3) for g in (0.5, 0.9)]}
NAMES = {"q": "Q-learning", "sarsa": "SARSA", "linucb": "LinUCB (contextual bandit)", "dqn": "Deep Q-network"}
LAM_U = [2.0, 4.0, 8.0, 16.0, 32.0]
SOURCES = {"vol": "Lab 7", "vix": "Lab 7", "mom": "Lab 7", "rates": "Lab 7", "credit": "Lab 7", "macro": "Lab 7",
           "termvol": "Cboe (Yahoo), FRED", "stress": "FRED (St. Louis Fed, Chicago Fed, EPU)",
           "cross": "Yahoo", "crossvol": "Yahoo, FRED (Cboe)", "factors": "Kenneth French library"}


def get_store() -> MarketStore:
    """The Lab 7 documents with the 30 new fields, in an in-memory store."""
    docs = MarketStore(STORE).t.all()
    idx = pd.DatetimeIndex(sorted(pd.Timestamp(d["date"]) for d in docs))
    store = MarketStore()
    store.ingest(X.extend_documents(docs, X.new_fields(idx), seed=C.SEED + 1))
    return store


def get_frame() -> pd.DataFrame:
    """All 50 features and the target as read back through the ingestion layer (cached)."""
    if FRAME.exists():
        return pd.read_parquet(FRAME)
    frame = X.read_frame(get_store(), X.features(X.GROUPS11))
    frame.to_parquet(FRAME)
    return frame


def get_tables():
    if TABLES.exists():
        return pickle.loads(TABLES.read_bytes())
    store = get_store()
    frame = X.read_frame(store, X.features(X.GROUPS11))
    frame.to_parquet(FRAME)
    dates = frame.index
    first = int(np.searchsorted(dates, pd.Timestamp("2003-10-01")))
    stress = frame["stlfsi"].ffill()
    out = {"n_docs": len(store.t), "n_field_names": len(store.schema_versions())}
    for key, groups in (("6", X.GROUPS6), ("11", X.GROUPS11)):
        t0 = time.time()
        out[key] = X.build_tables(frame, groups, first, len(dates) - 5, stress, workers=12)
        out[f"minutes_{key}"] = (time.time() - t0) / 60
        print(f"{time.strftime('%H:%M:%S')} tables {key} groups: {out[f'minutes_{key}']:.1f} min", flush=True)
    out["missing"] = frame[X.features(X.GROUPS_NEW)].notna().mean()
    out["first_valid"] = frame[X.features(X.GROUPS_NEW)].apply(lambda s: s.first_valid_index())
    TABLES.write_bytes(pickle.dumps(out))
    return out


def splits(T):
    """Purged training, validation and test decision points (as in Lab 7)."""
    return purged_blocks(T.dates, T.ends)


def make(kind, T, hp, seed):
    na, d = T.ng + 1, 4 + 2 * T.ng
    if kind in ("q", "sarsa", "random"):
        return X.Tabular(na, alpha=hp.get("alpha", 0.1), gamma=hp.get("gamma", 0.9), sarsa=kind == "sarsa", seed=seed,
                         eps=1.0 if kind == "random" else 0.5)
    if kind == "linucb":
        return X.LinUCB(na, d, alpha=hp["alpha"], seed=seed)
    return X.DQN(na, d, gamma=hp["gamma"], lr=hp["lr"], seed=seed)


_T = {}


def _init(tables):
    import torch
    torch.set_num_threads(1)
    _T.update(tables)


def job(spec):
    """Train on the training block, then learn online through validation (and test)."""
    kind, tkey, table, lam, hp, seed, phase = spec
    T = _T[tkey]
    rel = not table.endswith("_abs")                    # "rmse_abs": the absolute group cost of Lab 7
    table = table.removesuffix("_abs")
    Env = lambda T_, steps, table_, lam_: X.Env(T_, steps, table_, lam_, relative=rel)
    tr, va, te = splits(T)
    ag = make(kind, T, hp, seed)
    t0, curve = time.time(), []
    out = {}
    if kind == "random":
        e, m, _ = ag.episode(Env(T, te, table, lam), learn=False)
        return spec, {"test": X.pooled(e, T.nobs[te]), "errs": e, "masks": m}
    n = EPISODES[kind]
    rng = np.random.default_rng(seed)
    for ep in range(n):
        ag.eps = X.eps_schedule(ep, n)
        env = Env(T, tr, table, lam)
        _, _, rw = ag.episode(env, mask=env.random_mask(rng))            # exploring starts
        curve.append(rw.mean())
    ag.eps = 0.05 if kind != "linucb" else 0.0
    _, mtr, _ = ag.episode(Env(T, tr, table, lam))                     # a last pass from all groups
    ev, mva, _ = ag.episode(Env(T, va, table, lam), mask=mtr[-1])     # online: the subset carries over
    out.update(val=X.pooled(ev, T.nobs[va]), curve=np.array(curve))
    if phase == "test":
        e, m, _ = ag.episode(Env(T, te, table, lam), mask=mva[-1])
        out.update(test=X.pooled(e, T.nobs[te]), errs=e, masks=m)
    out["seconds"] = time.time() - t0
    if kind in ("q", "sarsa"):
        out["states"] = len(ag.Q)
    return spec, out


def run_jobs(specs, tables, workers=12):
    with ProcessPoolExecutor(workers, initializer=_init, initargs=(tables,)) as ex:
        return [out for _, out in ex.map(job, specs, chunksize=1)]


def get_runs(tabs):
    tables = {"6": tabs["6"], "11": tabs["11"]}
    if RUNS.exists():
        runs = pickle.loads(RUNS.read_bytes())
        if "abs" not in runs:
            add_absolute(runs, tables)
        return runs
    hp_key = lambda hp: tuple(sorted(hp.items()))
    runs = {"grid": {}, "final": {}, "unlab": {}}
    # ---------------- validation grids (3 seeds)
    specs = [(k, t, "rmse", None, hp, s, "val") for t in tables for k, hps in GRID.items() for hp in hps for s in range(3)]
    for (k, t, _, _, hp, _, _), o in zip(specs, run_jobs(specs, tables)):
        runs["grid"].setdefault((t, k, hp_key(hp)), []).append(o["val"])
    print(f"{time.strftime('%H:%M:%S')} grids done", flush=True)
    runs["best"] = {(t, k): min((hp for (tt, kk, hp) in runs["grid"] if tt == t and kk == k),
                                key=lambda hp: np.mean(runs["grid"][(t, k, hp)])) for t in tables for k in GRID}
    # ---------------- final runs (10 seeds), random policies and the unlabeled variant (10 groups)
    specs = [(k, t, "rmse", None, dict(runs["best"][(t, k)]), s, "test") for t in tables for k in GRID for s in range(10)]
    specs += [("random", t, "rmse", None, {}, s, "test") for t in tables for s in range(10)]
    specs += [("q", t, "recon", lam, {"alpha": 0.1, "gamma": 0.9}, s, "test") for t in tables for lam in LAM_U
              for s in range(10)]
    for (k, t, tb, lam, _, _, _), o in zip(specs, run_jobs(specs, tables)):
        if tb == "recon":
            runs["unlab"].setdefault((t, lam), []).append(o)
        else:
            runs["final"].setdefault((t, k), []).append(o)
    print(f"{time.strftime('%H:%M:%S')} final runs done", flush=True)
    add_absolute(runs, tables)
    return runs


def add_absolute(runs, tables):
    """The final runs repeated with the absolute group cost of Lab 7 (a shift of the reward by an
    action-independent amount, which leaves the optimal policy unchanged)."""
    specs = [(k, t, "rmse_abs", None, dict(runs["best"][(t, k)]), s, "test") for t in tables for k in GRID for s in range(10)]
    runs["abs"] = {}
    for (k, t, _, _, _, _, _), o in zip(specs, run_jobs(specs, tables)):
        runs["abs"].setdefault((t, k), []).append(o)
    print(f"{time.strftime('%H:%M:%S')} runs with the absolute cost done", flush=True)
    RUNS.write_bytes(pickle.dumps(runs))


def gname(mask, G):
    return "+".join(g for i, g in enumerate(G) if mask >> i & 1)


def baselines(T, te, trva):
    nobs = T.nobs[te]
    ALL = (1 << T.ng) - 1
    all_e = np.array([T.rmse[k, T.avail[k]] for k in te])
    fixed = {m: X.pooled(np.array([T.rmse[k, m] if not (m & ~T.avail[k]) else T.rmse[k, T.avail[k]] for k in trva]),
                         T.nobs[trva]) for m in range(1, ALL + 1)}
    m_fix = min(fixed, key=fixed.get)
    fix_e = np.array([T.rmse[k, m_fix] if not (m_fix & ~T.avail[k]) else T.rmse[k, T.avail[k]] for k in te])
    inner_m = [int(np.nanargmin(T.inner[k])) for k in te]
    inner_e = np.array([T.rmse[k, m] for k, m in zip(te, inner_m)])
    oracle_e = np.array([np.nanmin(T.rmse[k]) for k in te])
    return {"all": (X.pooled(all_e, nobs), T.ng, 0, all_e),
            "fixed": (X.pooled(fix_e, nobs), X.popcount(m_fix), 0, fix_e, m_fix),
            "inner": (X.pooled(inner_e, nobs), np.mean([X.popcount(m) for m in inner_m]),
                      int(np.sum(np.diff(inner_m) != 0)), inner_e),
            "lasso": (X.pooled(T.lasso[te], nobs), np.nan, np.nan, T.lasso[te]),
            "oracle": (X.pooled(oracle_e, nobs), np.nan, np.nan, oracle_e)}


def main():
    tabs = get_tables()
    T6, T10 = tabs["6"], tabs["11"]
    assert (T6.dates == T10.dates).all()
    tr, va, te = splits(T10)
    trva = np.r_[tr, va]
    runs = get_runs(tabs)
    G10 = T10.groups
    num = {"n_features": len(X.features(X.GROUPS11)), "n_new": len(X.features(X.GROUPS_NEW)), "n_groups": T10.ng,
           "n_subsets": 2 ** T10.ng - 1, "n_steps": len(T10.dates), "n_docs": tabs["n_docs"],
           "n_field_names": tabs["n_field_names"], "tables_minutes": tabs["minutes_6"] + tabs["minutes_11"],
           "n_configs": sum(len(v) for v in GRID.values()), "lam_u_ref": 8.0}

    # ---------------- table of the new groups
    rows = []
    for g in X.GROUPS_NEW:
        i = G10.index(g)
        first = T10.dates[np.argmax([(a >> i) & 1 for a in T10.avail])]
        fv = min(tabs["first_valid"][f] for f in X.GROUPS11[g])
        lv = max(tabs["first_valid"][f] for f in X.GROUPS11[g])
        rows.append((g, ", ".join(X.GROUPS11[g]).replace("_", "\\_"), SOURCES[g], lv.strftime("%Y-%m"),
                     first.strftime("%Y-%m"), 100 * np.mean([(a >> i) & 1 for a in T10.avail[te]])))
        num[f"first_{g}"] = first.strftime("%Y-%m-%d")
    save_table("lab7x_groups", pd.DataFrame(rows, columns=["Group", "Features", "Source", "Last series starts",
                                                           "First usable decision", "Available in test, \\%"]
                                            ).set_index("Group"),
               nd={"Available in test, \\%": 0}, col_format="lp{4.4cm}p{2.6cm}ccc")

    # ---------------- results: 6 vs 10 groups
    res_rows, curves = [], {}
    for t, T in (("6", T6), ("11", T10)):
        G = T.groups
        b = baselines(T, te, trva)
        nobs = T.nobs[te]
        rnd = [r["test"] for r in runs["final"][(t, "random")]]
        res = [("All available groups (fixed)", b["all"][0], np.nan, b["all"][1], b["all"][2]),
               (f"Best fixed subset on train+valid.", b["fixed"][0], np.nan, b["fixed"][1], 0),
               ("Re-selection at each step (inner validation)", b["inner"][0], np.nan, b["inner"][1], b["inner"][2]),
               ("Rolling lasso on all available features", b["lasso"][0], np.nan, np.nan, np.nan),
               ("Random policy (10 runs)", float(np.mean(rnd)), float(np.std(rnd)), np.nan, np.nan)]
        for k in GRID:
            rr = runs["final"][(t, k)]
            tests = [r["test"] for r in rr]
            res.append((f"{NAMES[k]} (10 seeds)", float(np.mean(tests)), float(np.std(tests)),
                        float(np.mean([[X.popcount(m) for m in r["masks"]] for r in rr])),
                        float(np.mean([np.sum(np.diff(r["masks"]) != 0) for r in rr]))))
            num[f"{k}_rmse_{t}"] = float(np.mean(tests)); num[f"{k}_sd_{t}"] = float(np.std(tests))
            num[f"{k}_seconds_{t}"] = float(np.mean([r["seconds"] for r in rr]))
            if k in ("q", "sarsa"):
                num[f"{k}_states_{t}"] = float(np.mean([r["states"] for r in rr]))
            if t == "11":
                curves[k] = np.array([r["curve"] for r in rr])
            num[f"{k}_hp_{t}"] = ", ".join(f"{a}={v:g}" for a, v in runs["best"][(t, k)])
        res.append(("Hindsight oracle (bound)", b["oracle"][0], np.nan, np.nan, np.nan))
        df = pd.DataFrame(res, columns=["Policy", "RMSE", "SD", "Groups", "Switches"]).set_index("Policy")
        res_rows.append(df)
        num.update({f"all_{t}": b["all"][0], f"fixed_{t}": b["fixed"][0], f"inner_{t}": b["inner"][0],
                    f"lasso_{t}": b["lasso"][0], f"random_{t}": float(np.mean(rnd)), f"oracle_{t}": b["oracle"][0],
                    f"fixed_mask_{t}": gname(b["fixed"][4], G)})
    both = pd.concat({"6 groups (20 features)": res_rows[0], "11 groups (50 features)": res_rows[1]}, axis=1)
    both = both.drop(columns=[("6 groups (20 features)", "Switches"), ("11 groups (50 features)", "Switches")])
    flat = both.copy()
    flat.columns = ["RMSE, 6", "SD, 6", "Groups, 6", "RMSE, 11", "SD, 11", "Groups, 11"]
    save_table("lab7x_results", flat, nd={"RMSE, 6": 4, "SD, 6": 4, "Groups, 6": 1, "RMSE, 11": 4, "SD, 11": 4,
                                          "Groups, 11": 1}, col_format="p{4.6cm}cccccc")
    print(flat.round(4).to_string())
    num["q_states_max_11"] = (2 ** T10.ng) * 6
    num["q_states_max_6"] = (2 ** T6.ng) * 6
    agents10 = {k: num[f"{k}_rmse_11"] for k in GRID}
    num["best_agent_11"] = NAMES[min(agents10, key=agents10.get)]
    num["best_agent_rmse_11"] = min(agents10.values())

    # ---------------- validation grid table
    grows = []
    for t in ("6", "11"):
        for k in GRID:
            for hp in GRID[k]:
                key = (t, k, tuple(sorted(hp.items())))
                v = runs["grid"][key]
                grows.append((t, NAMES[k], ", ".join(f"{a}={x:g}" for a, x in hp.items()), np.mean(v), np.std(v),
                              "*" if runs["best"][(t, k)] == key[2] else ""))
    gdf = pd.DataFrame(grows, columns=["Groups", "Agent", "Hyperparameters", "Valid. RMSE", "SD", "Chosen"])
    save_table("lab7x_grid", gdf.set_index("Groups"), nd={"Valid. RMSE": 4, "SD": 4}, col_format="llp{3.4cm}ccc")

    # ---------------- sensitivity to the zero point of the group cost
    zrows = []
    for k in GRID:
        row = [NAMES[k]]
        for t in ("6", "11"):
            a = [r["test"] for r in runs["final"][(t, k)]]; b = [r["test"] for r in runs["abs"][(t, k)]]
            row += [float(np.mean(a)), float(np.mean(b))]
            num[f"{k}_abs_{t}"] = float(np.mean(b))
        zrows.append(row)
    zdf = pd.DataFrame(zrows, columns=["Agent", "6 groups, relative", "6 groups, absolute", "11 groups, relative",
                                       "11 groups, absolute"]).set_index("Agent")
    save_table("lab7x_zero", zdf, nd=4, col_format="p{5.0cm}cccc")
    print(zdf.round(4).to_string())
    num["zero_max_change"] = float(np.max(np.abs(zdf.iloc[:, [0, 2]].to_numpy() - zdf.iloc[:, [1, 3]].to_numpy())))
    num["zero_max_change_6"] = float(np.max(np.abs(zdf.iloc[:, 0] - zdf.iloc[:, 1])))
    num["zero_max_change_11"] = float(np.max(np.abs(zdf.iloc[:, 2] - zdf.iloc[:, 3])))
    sd = {t: {k: np.std([r["test"] for r in runs["final"][(t, k)]]) for k in GRID} for t in ("6", "11")}
    num["sd_ratio_max"] = float(max(sd["11"][k] / sd["6"][k] for k in GRID))

    # ---------------- unlabeled variant: the best fixed subset under the label-free criterion and the agent
    urows, use_u, ucurves = [], [], {}
    for t, T in (("6", T6), ("11", T10)):
        nobs, ALL = T.nobs[te], (1 << T.ng) - 1
        for lam in LAM_U:
            rr = runs["unlab"][(t, lam)]
            fixed = {m: np.mean([X.Env.SCALE * T.recon[k, m] + lam * X.popcount(m) for k in trva if not (m & ~T.avail[k])])
                     for m in range(1, ALL + 1) if sum(1 for k in trva if not (m & ~T.avail[k])) > 0.5 * len(trva)}
            m_u = min(fixed, key=fixed.get)
            fe = np.array([T.rmse[k, m_u] if not (m_u & ~T.avail[k]) else T.rmse[k, T.avail[k]] for k in te])
            tests = [r["test"] for r in rr]
            urows.append((t, f"{lam:g}", gname(m_u, T.groups).replace("+", ", "), X.pooled(fe, nobs),
                          float(np.mean([[X.popcount(m) for m in r["masks"]] for r in rr])), float(np.mean(tests)),
                          float(np.std(tests))))
            ucurves.setdefault(t, []).append((lam, urows[-1][3], urows[-1][5]))
            if t == "11":
                use_u.append([np.mean([(m >> i) & 1 for r in rr for m in r["masks"]]) for i in range(T.ng)])
            key = f"{lam:g}_{t}"
            num.update({f"unlab_rmse_{key}": urows[-1][5], f"unlab_sd_{key}": urows[-1][6], f"unlab_groups_{key}": urows[-1][4],
                        f"unlab_fixed_{key}": urows[-1][3], f"unlab_fixed_mask_{key}": urows[-1][2]})
    udf = pd.DataFrame(urows, columns=["Groups", "$\\lambda_u$", "Best fixed subset without labels", "RMSE (fixed)",
                                       "Groups (agent)", "RMSE (agent)", "SD"]).set_index("Groups")
    save_table("lab7x_unlabeled", udf, nd={"RMSE (fixed)": 4, "Groups (agent)": 1, "RMSE (agent)": 4, "SD": 4},
               col_format="ccp{4.6cm}cccc")
    print(udf.round(4).to_string())
    u_all = np.array([T10.recon[k, T10.avail[k]] for k in te])
    num["recon_all"] = float(np.mean(u_all))
    num["recon_vol_vix"] = float(np.nanmean([T10.recon[k, 0b11] for k in te]))
    num["recon_single_med"] = float(np.nanmedian([T10.recon[k, 1 << i] for k in te for i in range(T10.ng)
                                                  if T10.avail[k] >> i & 1]))
    single = {g: float(np.nanmean([T10.recon[k, 1 << i] for k in te if T10.avail[k] >> i & 1])) for i, g in enumerate(G10)}
    num["recon_vix"] = single["vix"]
    num["recon_vix_rank"] = int(sorted(single, key=single.get).index("vix") + 1)
    num["recon_best_single"] = min(single, key=single.get)
    num["recon_second_single"] = sorted(single, key=single.get)[1]

    # ---------------- figures
    import matplotlib.dates as mdates
    f, ax = P.fig(2.6)
    av = np.array([[(a >> i) & 1 for a in T10.avail] for i in range(T10.ng)])
    ax.imshow(av, aspect="auto", cmap=P.SEQ, vmin=0, vmax=1.3, interpolation="nearest",
              extent=[mdates.date2num(T10.dates[0]), mdates.date2num(T10.dates[-1]), T10.ng - 0.5, -0.5])
    ax.xaxis_date(); ax.set_yticks(range(T10.ng)); ax.set_yticklabels(G10, fontsize=9); ax.grid(False)
    for d in (C.TRAIN_END, C.VAL_END):
        ax.axvline(mdates.date2num(pd.Timestamp(d)), color=P.RED, lw=0.9, ls="--")
    ax.axhline(5.5, color="white", lw=1.2)
    P.save(f, "lab7x_availability")

    f, ax = P.fig(3.6)
    names = list(res_rows[0].index)
    y = np.arange(len(names))
    for j, (df, col, lab) in enumerate(((res_rows[0], P.MUTED, "6 groups, 20 features"),
                                        (res_rows[1], P.BLUE, "11 groups, 50 features"))):
        ax.barh(y + (j - 0.5) * 0.38, df["RMSE"], height=0.36, color=col, label=lab,
                xerr=df["SD"].fillna(0), error_kw={"lw": 0.7, "capsize": 1.5})
    ax.set_yticks(y); ax.set_yticklabels(names, fontsize=9); ax.invert_yaxis()
    lo = min(res_rows[0]["RMSE"].min(), res_rows[1]["RMSE"].min())
    ax.set_xlim(lo - 0.03, max(res_rows[0]["RMSE"].max(), res_rows[1]["RMSE"].max()) + 0.02)
    ax.set_xlabel("test RMSE of the 5-day log volatility forecasts, 2019--2026"); ax.legend(fontsize=9, loc="lower right")
    f.tight_layout(); P.save(f, "lab7x_agents")

    f, ax = P.fig(2.6)
    for k, col in zip(("q", "sarsa", "dqn", "linucb"), (P.BLUE, P.ORANGE, P.AQUA, P.RED)):
        cv = curves[k]
        xs = np.linspace(0, 1, cv.shape[1])
        ax.plot(xs, cv.mean(0), color=col, lw=1.0, label=f"{NAMES[k]} ({cv.shape[1]} episodes)")
        ax.fill_between(xs, cv.mean(0) - cv.std(0), cv.mean(0) + cv.std(0), color=col, alpha=0.15, lw=0)
    ax.set_xlabel("share of the training episodes"); ax.set_ylabel("mean reward per step"); ax.legend(fontsize=9)
    P.save(f, "lab7x_learning")

    f, axs = P.fig(3.7, ncols=2, gridspec_kw={"width_ratios": [1, 1.5]})
    for t, col in (("6", P.ORANGE), ("11", P.BLUE)):
        c = np.array(ucurves[t])
        axs[0].plot(c[:, 0], c[:, 1], "s--", color=col, lw=1, ms=3, label=f"fixed subset, {t} groups")
        axs[0].plot(c[:, 0], c[:, 2], "o-", color=col, lw=1, ms=3, label=f"agent, {t} groups")
    axs[0].axhline(num["fixed_11"], color=P.INK2, lw=0.8, ls=":", label="best fixed subset, labels")
    axs[0].set_xscale("log", base=2); axs[0].set_xlabel("$\\lambda_u$"); axs[0].set_ylabel("test RMSE")
    axs[0].legend(fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.2))
    U = np.array(use_u)
    axs[1].imshow(U, cmap=P.SEQ_CELL, vmin=0, vmax=1, aspect="auto")
    for i in range(U.shape[0]):
        for j in range(U.shape[1]):
            axs[1].text(j, i, P.per_cent(U[i, j]), ha="center", va="center", fontsize=9, color=P.INK)
    axs[1].set_xticks(range(T10.ng)); axs[1].set_xticklabels(G10, rotation=60, fontsize=9)
    axs[1].set_yticks(range(len(LAM_U))); axs[1].set_yticklabels([f"{x:g}" for x in LAM_U], fontsize=9)
    axs[1].set_ylabel("$\\lambda_u$"); axs[1].grid(False)
    f.tight_layout(); P.save(f, "lab7x_unlabeled")

    f, ax = P.fig(2.4)
    use = np.array([[np.mean([(m >> i) & 1 for r in runs["final"][("11", k)] for m in r["masks"]]) for i in range(T10.ng)]
                    for k in GRID])
    ax.imshow(use, cmap=P.SEQ_CELL, vmin=0, vmax=1, aspect="auto")
    for i in range(use.shape[0]):
        for j in range(use.shape[1]):
            ax.text(j, i, P.per_cent(use[i, j]), ha="center", va="center", fontsize=9, color=P.INK)
    ax.set_xticks(range(T10.ng)); ax.set_xticklabels(G10, fontsize=9); ax.set_yticks(range(len(GRID)))
    ax.set_yticklabels([NAMES[k] for k in GRID], fontsize=9); ax.grid(False)
    P.save(f, "lab7x_usage")
    num.update({f"use_{k}_{g}": float(use[a, i]) for a, k in enumerate(GRID) for i, g in enumerate(G10)})
    save_numbers("lab7x", num)
    print(num)


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    main()
