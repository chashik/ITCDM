"""Lab 6.6. The same networks on two devices.

The five LSTMs of Lab 6.1 are trained with the same seeds, data and
mini-batches on the CPU and on the GPU.  The only difference is the order of
floating-point operations inside the kernels.  The script measures how far
this difference propagates through early stopping to the forecasts and to
the comparison with the HAR model.  Without a GPU it does nothing.

Run:  python -m lab6_rnn.run_devices
"""
from __future__ import annotations

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LinearRegression

from dmfin.device import overlap_allowed
from dmfin.reporting import save_numbers, save_table

from .common import lab6_frame, rmse
from .nets import INPUTS, SeqData, SeqNet, diebold_mariano, gpu_available, predict, set_seed, train
from .run_forecast import HAR, HIDDEN, HORIZON, LOOKBACK, SEEDS


def _job(args):
    device, seed = args
    torch.set_num_threads(2)
    df, _ = lab6_frame()
    sd = SeqData(df, "y_logrv_5", lookback=LOOKBACK, horizon=HORIZON)
    set_seed(seed)
    m = SeqNet(len(INPUTS), HIDDEN, 1, "lstm")
    m, hist = train(m, sd["train"].X, sd["train"].y, sd["val"].X, sd["val"].y, lr=1e-3, batch=64,
                    max_epochs=100, patience=10, weight_decay=1e-5, seed=seed, device=device)
    return device, seed, predict(m, sd["test"].X), hist


def main():
    if not gpu_available():
        print("no GPU: nothing to compare")
        return
    df, _ = lab6_frame()
    sd = SeqData(df, "y_logrv_5", lookback=LOOKBACK, horizon=HORIZON)
    yte, dte = sd["test"].y, sd["test"].dates
    trva = df.loc[sd["train"].dates.append(sd["val"].dates)]
    har = LinearRegression().fit(trva[HAR], trva["y_logrv_5"]).predict(df.loc[dte, HAR])
    with ProcessPoolExecutor(10, mp_context=mp.get_context("spawn")) as ex:
        if overlap_allowed():
            out = list(ex.map(_job, [(d, s) for d in ("cpu", "cuda") for s in SEEDS]))
        else:                                   # CPU and GPU one after the other (dmfin/device.py)
            out = list(ex.map(_job, [("cpu", s) for s in SEEDS])) + list(ex.map(_job, [("cuda", s) for s in SEEDS]))
    res = {d: {s: (p, h) for dd, s, p, h in out if dd == d} for d in ("cpu", "cuda")}
    rows, num = [], {}
    for d, name in (("cpu", "CPU"), ("cuda", "GPU")):
        preds = [res[d][s][0] for s in SEEDS]
        ens = np.mean(preds, axis=0)
        stat, p = diebold_mariano(yte - ens, yte - har, h=HORIZON)
        best_ep = [int(np.argmin(res[d][s][1][:, 1])) + 1 for s in SEEDS]
        rows.append((name, np.mean([rmse(yte, q) for q in preds]), rmse(yte, ens), stat, p,
                     " ".join(str(e) for e in best_ep)))
        num.update({f"{d}_ens": rmse(yte, ens), f"{d}_dm": stat, f"{d}_p": p})
    tab = pd.DataFrame(rows, columns=["Device", "Test RMSE (single, mean)", "Test RMSE (ensemble)", "DM vs HAR",
                                      "$p$", "Best epoch of seeds 0--4"]).set_index("Device")
    save_table("lab6_devices", tab, nd={"Test RMSE (single, mean)": 4, "Test RMSE (ensemble)": 4, "DM vs HAR": 2,
                                        "$p$": 3}, col_format="lccccc")
    print(tab.to_string())
    diff_first = max(abs(res["cpu"][s][1][0, 1] - res["cuda"][s][1][0, 1]) for s in SEEDS)
    same_stop = sum(int(np.argmin(res["cpu"][s][1][:, 1])) == int(np.argmin(res["cuda"][s][1][:, 1])) for s in SEEDS)
    num.update({"diff_first_epoch": diff_first, "same_stop": same_stop, "n_seeds": len(SEEDS),
                "max_pred_diff": float(max(np.abs(res["cpu"][s][0] - res["cuda"][s][0]).max() for s in SEEDS)),
                "har_test": rmse(yte, har)})
    save_numbers("lab6d", num)
    print(num)


if __name__ == "__main__":
    main()
