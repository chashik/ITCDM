"""Lab 7. Reinforcement-learning feature selection on a volatile data stream.

Run:  python -m lab7_rl_features.run
"""
from __future__ import annotations

import pickle

import numpy as np
import pandas as pd

from dmfin import config as C
from dmfin import plotting as P
from dmfin.data import load_market_panel
from dmfin.features import add_targets, market_features
from dmfin.reporting import save_numbers, save_table

from .env import (ALL, G, KEEP, NG, FeatureSelectionEnv, QAgent, build_tables, pooled, popcount, purged_blocks,
                  regime, run_episode)
from .stream import FEATURES, MarketStore, canonical_frame, to_documents

TABLES = C.RESULTS_DIR / "lab7_tables.pkl"
EPISODES = 300


def get_tables():
    frame = canonical_frame(add_targets(market_features(load_market_panel()), horizons=(5,)))
    frame = frame.loc["2000-06-01":]
    store = MarketStore(C.RESULTS_DIR / "lab7_stream.json")
    if TABLES.exists():
        return store, pickle.loads(TABLES.read_bytes())
    store.ingest(to_documents(frame, seed=C.SEED))
    dates = frame.index
    first = int(np.searchsorted(dates, pd.Timestamp("2003-10-01")))
    T = build_tables(store, dates, first, len(dates) - 5)
    TABLES.write_bytes(pickle.dumps(T))
    return store, T


def train_agent(T, tr_steps, alpha, gamma, seed, episodes=EPISODES):
    ag = QAgent(alpha=alpha, gamma=gamma, eps=0.5, seed=seed)
    env = FeatureSelectionEnv(T, tr_steps)
    curve = []
    for ep in range(episodes):
        ag.eps = max(0.05, 0.5 * (1 - ep / (0.8 * episodes)))
        _, _, rw = run_episode(env, ag, learn=True)
        curve.append(rw.mean())
    ag.eps = 0.05
    return ag, np.array(curve)


def main():
    store, T = get_tables()
    num = {"n_steps": len(T.dates), "first_decision": T.dates[0].strftime("%Y-%m-%d"),
           "n_docs": len(store.t), "lambda": FeatureSelectionEnv.LAMBDA, "psi": FeatureSelectionEnv.PSI,
           "episodes": EPISODES, "n_features": len(FEATURES)}
    sv = store.schema_versions()
    num.update({"n_field_names": len(sv)})

    tr, va, te = purged_blocks(T.dates, T.ends)
    num.update({"n_tr_steps": len(tr), "n_va_steps": len(va), "n_te_steps": len(te),
                "n_purged_steps": len(T.dates) - len(tr) - len(va) - len(te)})

    # --------------------------------------------- hyperparameters on validation
    grid = []
    for alpha in (0.05, 0.1, 0.3):
        for gamma in (0.0, 0.5, 0.9):
            vals = []
            for seed in range(5):
                ag, _ = train_agent(T, tr, alpha, gamma, seed, episodes=150)
                e, _, _ = run_episode(FeatureSelectionEnv(T, va), ag, learn=True)
                vals.append(pooled(e, T.nobs[va]))
            grid.append((alpha, gamma, np.mean(vals), np.std(vals)))
    gdf = pd.DataFrame(grid, columns=["$\\alpha$", "$\\gamma$", "Valid. RMSE", "SD"])
    save_table("lab7_grid", gdf.set_index("$\\alpha$"), nd={"$\\gamma$": 1, "Valid. RMSE": 4, "SD": 4})
    best = gdf.loc[gdf["Valid. RMSE"].idxmin()]
    alpha, gamma = float(best["$\\alpha$"]), float(best["$\\gamma$"])
    num.update({"alpha": alpha, "gamma": gamma})

    # ------------------------------------------------ final runs (10 seeds)
    trva = np.r_[tr, va]
    rl_err, rl_masks, curves = [], [], []
    for seed in range(10):
        ag, curve = train_agent(T, tr, alpha, gamma, seed)
        curves.append(curve)
        run_episode(FeatureSelectionEnv(T, va), ag, learn=True)          # continue online through validation
        e, m, _ = run_episode(FeatureSelectionEnv(T, te), ag, learn=True)  # and through the test period
        rl_err.append(e); rl_masks.append(m)
    rl_err = np.array(rl_err); rl_masks = np.array(rl_masks)

    nobs = T.nobs[te]
    all_e = np.array([T.rmse[k, T.avail[k]] for k in te])
    fixed_scores = {m: pooled(np.array([T.rmse[k, m] if not (m & ~T.avail[k]) else T.rmse[k, T.avail[k]]
                                        for k in trva]), T.nobs[trva]) for m in range(1, ALL + 1)}
    m_fix = min(fixed_scores, key=fixed_scores.get)
    fix_e = np.array([T.rmse[k, m_fix] for k in te])
    inner_m = [int(np.nanargmin(T.inner[k])) for k in te]
    inner_e = np.array([T.rmse[k, m] for k, m in zip(te, inner_m)])
    oracle_e = np.array([np.nanmin(T.rmse[k]) for k in te])
    rng = np.random.default_rng(C.SEED)
    rnd = []
    for seed in range(10):
        ag = QAgent(eps=1.0, seed=seed)
        e, _, _ = run_episode(FeatureSelectionEnv(T, te), ag, learn=False)
        rnd.append(pooled(e, nobs))
    rows = [("All available groups (fixed)", pooled(all_e, nobs), NG, 0),
            (f"Best fixed subset on train+valid. ({'+'.join(g for i, g in enumerate(G) if m_fix >> i & 1)})",
             pooled(fix_e, nobs), popcount(m_fix), 0),
            ("Best subset re-selected at each step (inner validation)", pooled(inner_e, nobs),
             np.mean([popcount(m) for m in inner_m]), int(np.sum(np.diff(inner_m) != 0))),
            ("Rolling lasso, $\\alpha$ by inner validation", pooled(T.lasso[te], nobs), np.nan, np.nan),
            ("Random policy (mean of 10 runs)", float(np.mean(rnd)), np.nan, np.nan),
            ("Q-learning agent (mean of 10 seeds)", float(np.mean([pooled(e, nobs) for e in rl_err])),
             float(np.mean([[popcount(m) for m in ms] for ms in rl_masks])),
             float(np.mean([np.sum(np.diff(ms) != 0) for ms in rl_masks]))),
            ("Hindsight oracle (upper bound)", pooled(oracle_e, nobs), np.nan, np.nan)]
    res = pd.DataFrame(rows, columns=["Policy", "Test RMSE", "Mean active groups", "Switches"]).set_index("Policy")
    save_table("lab7_results", res, nd={"Test RMSE": 4, "Mean active groups": 1, "Switches": 0},
               col_format="p{8.2cm}ccc")
    print(res.round(4).to_string())
    num.update({"rmse_all": res.iloc[0, 0], "rmse_fixed": res.iloc[1, 0], "rmse_inner": res.iloc[2, 0],
                "rmse_lasso": res.iloc[3, 0], "rmse_random": res.iloc[4, 0], "rmse_rl": res.iloc[5, 0],
                "rmse_oracle": res.iloc[6, 0], "rl_groups": res.iloc[5, 1], "rl_switches": res.iloc[5, 2],
                "inner_switches": res.iloc[2, 2], "rl_sd": float(np.std([pooled(e, nobs) for e in rl_err])),
                "fixed_mask": "+".join(g for i, g in enumerate(G) if m_fix >> i & 1),
                "gain_rl_vs_all": 100 * (1 - res.iloc[5, 0] / res.iloc[0, 0])})
    # availability of groups over time
    num["macro_first"] = T.dates[np.argmax([(a >> G.index("macro")) & 1 for a in T.avail])].strftime("%Y-%m-%d")

    # ------------------------------------------------------------- figures
    f, ax = P.fig(2.8)
    show = sv.loc[[n for n in sv.index if n in ("VIX", "log_vix", "term_bp", "term_10y3m", "breakeven_10y",
                                                 "usd_ret_22", "ret_66", "volume_z")]]
    ax.imshow(show.to_numpy().astype(float), cmap=P.SEQ, vmin=0, vmax=1.4, aspect="auto")
    ax.set_yticks(range(len(show))); ax.set_yticklabels(show.index, fontsize=9)
    ax.set_xticks(range(0, show.shape[1], 2)); ax.set_xticklabels(show.columns[::2], fontsize=9, rotation=45)
    ax.grid(False)
    P.save(f, "lab7_schema")

    f, ax = P.fig(2.4)
    cv = np.array(curves)
    ax.plot(np.arange(1, cv.shape[1] + 1), cv.mean(0), color=P.BLUE, lw=1.0)
    ax.fill_between(np.arange(1, cv.shape[1] + 1), cv.mean(0) - cv.std(0), cv.mean(0) + cv.std(0),
                    color=P.BLUE, alpha=0.2, lw=0)
    ax.set_xlabel("training episode (one pass over 2003-2014)"); ax.set_ylabel("mean reward per step")
    P.save(f, "lab7_learning")

    dte = T.dates[te]
    f, axs = P.fig(4.2, nrows=2, sharex=True, gridspec_kw={"height_ratios": [1.3, 1]})
    base2 = all_e ** 2 * nobs
    for e, col, lab in [(inner_e, P.ORANGE, "re-selection by inner validation"), (T.lasso[te], P.AQUA, "rolling lasso"),
                        (rl_err.mean(0), P.BLUE, "Q-learning (seed average of errors)"), (fix_e, P.MUTED, "best fixed subset")]:
        axs[0].plot(dte, np.cumsum(base2 - e ** 2 * nobs), color=col, lw=1.1, label=lab)
    axs[0].axhline(0, color=P.INK2, lw=0.6)
    axs[0].legend(fontsize=9, loc="lower right", bbox_to_anchor=(1.0, 0.06), framealpha=0.95)
    axs[0].set_ylabel("cumulative SSE reduction\nvs all groups")
    usage = np.array([[ms[k] >> i & 1 for k in range(len(te))] for i in range(NG) for ms in rl_masks[:1]]).reshape(NG, -1)
    import matplotlib.dates as mdates
    axs[1].imshow(usage, aspect="auto", cmap=P.SEQ, vmin=0, vmax=1.3, interpolation="nearest",
                  extent=[mdates.date2num(dte[0]), mdates.date2num(dte[-1]), NG - 0.5, -0.5])
    axs[1].set_yticks(range(NG)); axs[1].set_yticklabels(G, fontsize=9); axs[1].grid(False)
    axs[1].set_ylabel("groups used\n(seed 0)")
    f.tight_layout(); P.save(f, "lab7_test")

    reg = np.array([regime(T.vix[k], T.vix_chg[k]) for k in te])
    names = ["VIX<16, falling", "VIX<16, rising", "16-25, falling", "16-25, rising", ">25, falling", ">25, rising"]
    freq = np.full((6, NG), np.nan)
    for r in range(6):
        if (reg == r).any():
            freq[r] = [np.mean([(m >> i) & 1 for ms in rl_masks for m in ms[reg == r]]) for i in range(NG)]
    f, ax = P.fig(2.8, w=4.8)
    ax.imshow(np.nan_to_num(freq), cmap=P.SEQ_CELL, vmin=0, vmax=1, aspect="auto")
    for r in range(6):
        for i in range(NG):
            if not np.isnan(freq[r, i]):
                ax.text(i, r, P.per_cent(freq[r, i]), ha="center", va="center", fontsize=9, color=P.INK)
    ax.set_xticks(range(NG)); ax.set_xticklabels(G); ax.set_yticks(range(6))
    ax.set_yticklabels([f"{n} ({(reg == r).sum()})" for r, n in enumerate(names)], fontsize=9); ax.grid(False)
    P.save(f, "lab7_usage")
    num.update({f"use_{g}": float(np.mean([(m >> i) & 1 for ms in rl_masks for m in ms])) for i, g in enumerate(G)})
    save_numbers("lab7", num)
    print(gdf.round(4).to_string()); print(num)


if __name__ == "__main__":
    main()
