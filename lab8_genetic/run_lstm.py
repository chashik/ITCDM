"""Lab 8.1. Genetic search over architecture, inputs and training of the
Lab 6 recurrent volatility forecaster; random search with the same budget
is the control.

Devices: a recurrent network is trained step by step along the sequence, so
on a GPU its time per epoch is dominated by kernel launches and hardly
depends on the width.  Narrow networks (hidden < 64) are therefore trained
on CPU cores, wide ones on the GPU, if there is one (DMFIN_DEVICE=cpu
disables it).  The rule depends on the genome only, so a rerun on the same
hardware reproduces every fitness value.

Run:  python -m lab8_genetic.run_lstm
"""
from __future__ import annotations

import multiprocessing as mp
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import torch

from dmfin import config as C
from dmfin import plotting as P
from dmfin.reporting import save_numbers, save_table
from lab6_rnn.common import lab6_frame, r2, rmse
from lab6_rnn.nets import (INPUTS, SeqData, SeqNet, diebold_mariano, gpu_available, gpu_rnn_check, predict,
                           set_seed, train)

from .ga import GeneticAlgorithm

SPACE = {
    "cell": ["rnn", "lstm", "gru"],
    "hidden": [8, 16, 32, 64, 128],
    "layers": [1, 2],
    "lookback": [10, 20, 40, 60, 120],
    "dropout": [0.0, 0.1, 0.2, 0.3],
    "lr": [3e-4, 1e-3, 3e-3],
    "batch": [32, 64, 128],
    "wd": [0.0, 1e-5, 1e-4],
}
KEYS = list(SPACE)
CARD = [len(v) for v in SPACE.values()] + [2] * len(INPUTS)     # + input mask bits
POP, GENS = 20, 10
from dmfin.device import GPU_HIDDEN, overlap_allowed, rnn_device  # noqa: E402  wide networks go to the GPU
CPU_WORKERS, GPU_WORKERS = 14, 6         # > 8 processes per GPU oversubscribe its queues
# without a GPU: one process per two hardware threads (each process uses two threads); DMFIN_WORKERS overrides
CPU_ONLY_WORKERS = int(os.environ.get("DMFIN_WORKERS", max(1, (os.cpu_count() or 2) // 2)))


def decode(g):
    cfg = {k: SPACE[k][g[i]] for i, k in enumerate(KEYS)}
    cfg["inputs"] = [c for c, b in zip(INPUTS, g[len(KEYS):]) if b]
    return cfg


def repair(g):
    g = list(g)
    if not any(g[len(KEYS):]):
        g[len(KEYS) + 1] = 1                     # keep at least log_pk_d
    return tuple(g)


def device_for(cfg, gpu: bool) -> str:
    return rnn_device(cfg["hidden"], gpu)


def fit_config(cfg, seed=0, device="cpu"):
    torch.set_num_threads(2)
    df, _ = lab6_frame()
    sd = SeqData(df, "y_logrv_5", lookback=cfg["lookback"], horizon=5, inputs=cfg["inputs"])
    set_seed(seed)                     # initial weights depend on the seed only, not on the worker
    m = SeqNet(len(cfg["inputs"]), cfg["hidden"], cfg["layers"], cfg["cell"], dropout=cfg["dropout"])
    m, hist = train(m, sd["train"].X, sd["train"].y, sd["val"].X, sd["val"].y, lr=cfg["lr"], batch=cfg["batch"],
                    max_epochs=100, patience=10, weight_decay=cfg["wd"], seed=seed, device=device)
    return m, sd, float(hist[:, 1].min())


def _fitness(args):
    g, device = args
    return fit_config(decode(g), device=device)[2]


def _final(args):
    cfg, seed, device = args
    m, sd, v = fit_config(cfg, seed, device)
    return predict(m, sd["val"].X), predict(m, sd["test"].X), sd["test"].dates, sd["val"].dates


class Trainer:
    """Two process pools, CPU and GPU; each job goes to the pool chosen by ``device_for``;
    the two pools work one after the other unless DMFIN_OVERLAP=1.

    Workers are started with 'spawn': a process that has initialised the GPU
    runtime must not be forked.
    """

    def __init__(self):
        self.gpu = gpu_available()
        if self.gpu and (err := gpu_rnn_check()):
            print(f"GPU disabled, recurrent networks with dropout fail on it: {err}\n"
                  "(ROCm: install the rocRAND headers, e.g. sudo apt install librocrand-dev)")
            self.gpu = False
        ctx = mp.get_context("spawn")
        self.pools = {"cpu": ProcessPoolExecutor(CPU_WORKERS if self.gpu else CPU_ONLY_WORKERS, mp_context=ctx)}
        if self.gpu:
            self.pools["cuda"] = ProcessPoolExecutor(GPU_WORKERS, mp_context=ctx)
        self.n_gpu = 0

    def map(self, fn, cfgs, args):
        devs = [device_for(c, self.gpu) for c in cfgs]
        self.n_gpu += devs.count("cuda")
        # longest jobs first (results do not depend on the order)
        cost = [c["hidden"] ** 2 * c["layers"] * c["lookback"] for c in cfgs]
        order = sorted(range(len(cfgs)), key=lambda i: -cost[i])
        # without DMFIN_OVERLAP=1 the GPU jobs run first and the CPU jobs after them (dmfin/device.py)
        phases = [order] if overlap_allowed() else [[i for i in order if devs[i] == d] for d in ("cuda", "cpu")]
        futs = {}
        for phase in phases:
            for i in phase:
                futs[i] = self.pools[devs[i]].submit(fn, (*args[i], devs[i]))
            for i in phase:
                futs[i].result()
        return [futs[i].result() for i in range(len(cfgs))]

    def shutdown(self):
        for p in self.pools.values():
            p.shutdown()


def main():
    t0 = time.time()
    pool = Trainer()
    batch = lambda gs: pool.map(_fitness, [decode(g) for g in gs], [(g,) for g in gs])
    default = tuple([SPACE["cell"].index("lstm"), SPACE["hidden"].index(32), 0, SPACE["lookback"].index(60),
                     0, SPACE["lr"].index(1e-3), SPACE["batch"].index(64), SPACE["wd"].index(1e-5)] + [1] * len(INPUTS))
    ga = GeneticAlgorithm(CARD, batch_fitness=batch, pop_size=POP, generations=GENS, tournament=3, p_cx=0.9,
                          p_mut=1.5 / len(CARD), elite=2, seed=C.SEED, repair=repair, init=[default])
    res = ga.run()
    budget = res.evaluations
    rng = np.random.default_rng(C.SEED + 1)
    rs = [repair(tuple(int(rng.integers(c)) for c in CARD)) for _ in range(budget)]
    rs_fit = batch(rs)
    rs_best = rs[int(np.argmin(rs_fit))]
    rs_curve = np.minimum.accumulate(rs_fit)
    default_fit = res.cache[default]

    cfgs = {"Lab 6 default LSTM": decode(default), "Random search best": decode(rs_best), "GA best": decode(res.best)}
    jobs = [(c, s) for c in cfgs.values() for s in range(5)]
    outs = pool.map(_final, [c for c, _ in jobs], jobs)
    pool.shutdown()
    df, _ = lab6_frame()
    rows, preds = [], {}
    for j, name in enumerate(cfgs):
        o = outs[5 * j:5 * j + 5]
        dte = o[0][2]
        yte = df.loc[dte, "y_logrv_5"].to_numpy()
        yva = df.loc[o[0][3], "y_logrv_5"].to_numpy()
        pt = np.mean([x[1] for x in o], axis=0); pv = np.mean([x[0] for x in o], axis=0)
        preds[name] = (dte, yte, pt)
        cfg = cfgs[name]
        rows.append((name, cfg["cell"].upper(), cfg["hidden"], cfg["layers"], cfg["lookback"], cfg["dropout"],
                     cfg["lr"], len(cfg["inputs"]), rmse(yva, pv), np.mean([rmse(yte, x[1]) for x in o]), rmse(yte, pt)))
    tab = pd.DataFrame(rows, columns=["Configuration", "Cell", "Hidden", "Layers", "Lookback", "Dropout", "LR",
                                      "Inputs", "Valid. RMSE", "Test RMSE (single)", "Test RMSE (ens.)"]).set_index("Configuration")
    save_table("lab8_lstm", tab, nd={"Hidden": 0, "Layers": 0, "Lookback": 0, "Dropout": 1, "LR": 4, "Inputs": 0,
                                     "Valid. RMSE": 3, "Test RMSE (single)": 3, "Test RMSE (ens.)": 3},
               col_format="p{3.0cm}cccccccccc")
    print(tab.to_string())
    # common test dates (lookbacks differ): compare on the intersection
    common = preds["GA best"][0].intersection(preds["Lab 6 default LSTM"][0])
    sel = lambda n: pd.Series(preds[n][2], index=preds[n][0]).loc[common].to_numpy()
    y = df.loc[common, "y_logrv_5"].to_numpy()
    dm, p = diebold_mariano(y - sel("GA best"), y - sel("Lab 6 default LSTM"), h=5)
    num = {"pop": POP, "gens": GENS, "evals": budget, "space_size": f"{int(np.prod(CARD)):,}".replace(",", "\\,"),
           "default_val_mse": default_fit, "ga_val_mse": res.best_fit, "rs_val_mse": float(min(rs_fit)),
           "dm_ga_default": dm, "p_ga_default": p,
           "ga_inputs": ", ".join(decode(res.best)["inputs"]).replace("_", "\\_"),
           "rs_inputs": ", ".join(decode(rs_best)["inputs"]).replace("_", "\\_"),
           **{f"{k}_{c}": v for k, c in [("ga", "test"), ("rs", "test"), ("def", "test")]
              for v in [tab.loc[{"ga": "GA best", "rs": "Random search best", "def": "Lab 6 default LSTM"}[k],
                                "Test RMSE (ens.)"]]}}
    num.update({"n_gpu": pool.n_gpu, "n_trained": budget * 2 + len(jobs), "gpu_used": "yes" if pool.gpu else "no",
                "minutes": (time.time() - t0) / 60})
    h = np.array(res.history)
    pd.to_pickle({"history": res.history, "rs_fit": list(rs_fit), "default_fit": default_fit,
                  "cache": {g: v for g, v in res.cache.items()}}, C.RESULTS_DIR / "lab8l_search.pkl")
    num.update({"div_start": h[0, 4], "div_end": h[-1, 4], "ga_evals_to_beat_rs":
                int(h[np.argmax(h[:, 1] <= min(rs_fit)), 5]) if (h[:, 1] <= min(rs_fit)).any() else -1})

    f, axs = P.fig(2.8, ncols=2)
    axs[0].plot(h[:, 5], h[:, 1], "o-", ms=3, color=P.BLUE, label="GA: best so far")
    axs[0].plot(h[:, 5], h[:, 3], "s--", ms=3, color=P.AQUA, label="GA: population median")
    axs[0].plot(np.arange(1, budget + 1), rs_curve, color=P.ORANGE, lw=1.1, label="random search: best so far")
    axs[0].axhline(default_fit, color=P.MUTED, ls=":", lw=1, label="Lab 6 default")
    axs[0].set_xlabel("number of trained networks"); axs[0].set_ylabel("validation MSE")
    axs[0].legend(fontsize=7); axs[0].set_ylim(min(h[:, 1].min(), rs_curve.min()) - 0.003, default_fit + 0.02)
    axs[1].plot(h[:, 0], h[:, 4], "o-", ms=3, color=P.BLUE)
    axs[1].set_xlabel("generation"); axs[1].set_ylabel("mean pairwise Hamming distance")
    f.tight_layout(); P.save(f, "lab8_lstm_search")

    allg = pd.DataFrame([{**decode(g), "fit": v} for g, v in res.cache.items()])
    f, axs = P.fig(2.5, ncols=3, sharey=True)
    for ax, k in zip(axs, ["cell", "lookback", "hidden"]):
        grp = allg.groupby(k)["fit"]
        xs = list(grp.groups)
        ax.boxplot([grp.get_group(x).values for x in xs], tick_labels=[str(x) for x in xs], widths=0.6,
                   medianprops=dict(color=P.ORANGE), boxprops=dict(color=P.BLUE), whiskerprops=dict(color=P.BLUE),
                   capprops=dict(color=P.BLUE), flierprops=dict(markersize=2, markeredgecolor=P.MUTED))
        ax.set_xlabel(k)
    axs[0].set_ylabel("validation MSE"); axs[0].set_ylim(allg["fit"].quantile(0.0) - 0.005, allg["fit"].quantile(0.95))
    f.tight_layout(); P.save(f, "lab8_lstm_genes")
    save_numbers("lab8l", num)
    print(num)


if __name__ == "__main__":
    main()
