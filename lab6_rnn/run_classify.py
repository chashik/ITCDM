"""Lab 6.2. Sequence classification: direction vs volatility regime.

Two binary targets for the next 5 trading days:
  * up    : cumulative log-return > 0;
  * stress: realised volatility above the 80th percentile of the training block.

Run:  python -m lab6_rnn.run_classify
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score, roc_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from dmfin import config as C
from dmfin import plotting as P
from dmfin.reporting import save_numbers, save_table

from .common import lab6_frame
from .nets import INPUTS, SeqData, SeqNet, predict, set_seed, train

LOOKBACK, H = 60, 5
BASE = ["har_pk_d", "har_pk_w", "har_pk_m", "log_vix", "ret", "term_10y3m", "credit_baa_aaa"]


def targets(df):
    tr_mask = df.index <= pd.Timestamp(C.TRAIN_END)
    q80 = df.loc[tr_mask, "y_rv_5"].quantile(0.8)
    d = df.copy()
    d["up"] = (d["y_ret_5"] > 0).astype(float)
    d["stress"] = (d["y_rv_5"] > q80).astype(float)
    return d, q80


def _job(args):
    task, seed = args
    torch.set_num_threads(2)
    df, _ = lab6_frame()
    d, _ = targets(df)
    sd = SeqData(d, task, lookback=LOOKBACK, horizon=H)
    set_seed(seed)                     # initial weights depend on the seed only
    m = SeqNet(len(INPUTS), 32, 1, "lstm", dropout=0.1)
    m, hist = train(m, sd["train"].X, sd["train"].y, sd["val"].X, sd["val"].y, loss="bce", lr=1e-3,
                    max_epochs=100, patience=10, weight_decay=1e-5, seed=seed)
    sig = lambda z: 1 / (1 + np.exp(-z))
    return task, seed, sig(predict(m, sd["val"].X)), sig(predict(m, sd["test"].X)), len(hist)


def main():
    df, _ = lab6_frame()
    d, q80 = targets(df)
    num = {"q80": 100 * q80}
    with ProcessPoolExecutor(10) as ex:
        out = list(ex.map(_job, [(t, s) for t in ("up", "stress") for s in range(5)]))
    rows, curves = [], {}
    for task, label in [("up", "Direction (up/down)"), ("stress", "High-volatility regime")]:
        sd = SeqData(d, task, lookback=LOOKBACK, horizon=H)
        yte, dte = sd["test"].y, sd["test"].dates
        pt = np.mean([o[3] for o in out if o[0] == task], axis=0)
        tr = d.loc[sd["train"].dates.append(sd["val"].dates)]
        lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(tr[BASE], tr[task])
        pl = lr.predict_proba(d.loc[dte, BASE])[:, 1]
        base_rate = yte.mean()
        clim = np.full_like(pl, tr[task].mean())
        for name, p in [("Logistic regression", pl), ("LSTM, 5-seed ensemble", pt)]:
            rows.append((label, name, 100 * base_rate, 100 * accuracy_score(yte, p >= 0.5),
                         roc_auc_score(yte, p), brier_score_loss(yte, p), brier_score_loss(yte, clim)))
        curves[task] = (yte, pl, pt, dte)
        num.update({f"{task}_rate": 100 * base_rate, f"{task}_auc_lstm": roc_auc_score(yte, pt),
                    f"{task}_auc_lr": roc_auc_score(yte, pl), f"{task}_acc_lstm": 100 * accuracy_score(yte, pt >= 0.5),
                    f"{task}_brier_lstm": brier_score_loss(yte, pt), f"{task}_brier_clim": brier_score_loss(yte, clim),
                    f"{task}_brier_lr": brier_score_loss(yte, pl)})
    tab = pd.DataFrame(rows, columns=["Task", "Model", "Positives, %", "Accuracy, %", "ROC AUC",
                                      "Brier", "Brier (climatology)"]).set_index("Task")
    save_table("lab6_classify", tab, nd={"Model": 0, "Positives, %": 1, "Accuracy, %": 1, "ROC AUC": 3,
                                         "Brier": 3, "Brier (climatology)": 3}, col_format="p{2.6cm}p{3.0cm}ccccc")
    print(tab.round(3).to_string())

    f, axs = P.fig(2.9, ncols=2)
    for ax, task, ttl in [(axs[0], "up", "direction"), (axs[1], "stress", "high-volatility regime")]:
        y, pl, pt, _ = curves[task]
        for p, col, nm in [(pl, P.ORANGE, "logistic"), (pt, P.BLUE, "LSTM")]:
            fpr, tpr, _ = roc_curve(y, p)
            ax.plot(fpr, tpr, color=col, lw=1.1, label=f"{nm} (AUC {roc_auc_score(y, p):.3f})")
        ax.plot([0, 1], [0, 1], color=P.MUTED, ls="--", lw=0.8)
        ax.set_title(ttl, fontsize=9); ax.set_xlabel("false positive rate"); ax.legend(fontsize=9, loc="lower right")
    axs[0].set_ylabel("true positive rate")
    f.tight_layout(); P.save(f, "lab6_classify_roc")

    y, pl, pt, dte = curves["stress"]
    f, ax = P.fig(2.6)
    ax.fill_between(dte, 0, y, step="post", color="#f2d7d5", lw=0, label="actual stress week")
    ax.plot(dte, pt, color=P.BLUE, lw=0.8, label="LSTM probability")
    ax.axhline(0.5, color=P.INK2, lw=0.6, ls="--")
    ax.set_ylabel("P(stress next week)"); ax.set_ylim(0, 1.02); ax.legend(loc="upper right", fontsize=9)
    P.save(f, "lab6_stress_prob")
    save_numbers("lab6c", num)
    print(num)


if __name__ == "__main__":
    main()
