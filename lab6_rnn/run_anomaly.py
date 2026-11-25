"""Lab 6.3. Anomaly detection with an LSTM autoencoder.

The autoencoder learns to reconstruct 20-day windows of the 8 standardised
market channels.  Windows it cannot reconstruct well are anomalous.  The
threshold is the 99th percentile of reconstruction errors in the
validation block; detected test windows are compared with a classical
univariate rule (|return| > 4 rolling standard deviations).

Run:  python -m lab6_rnn.run_anomaly
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from dmfin import config as C
from dmfin import plotting as P
from dmfin.reporting import save_numbers, save_table

from .common import lab6_frame
from .nets import INPUTS, LSTMAutoencoder, SeqData, recon_error, train

WIN = 20
torch.set_num_threads(16)


def episodes(dates: pd.DatetimeIndex, gap_days: int = 10):
    """Merge flagged dates into episodes separated by > gap_days calendar days."""
    if len(dates) == 0:
        return []
    eps, start, prev = [], dates[0], dates[0]
    for d in dates[1:]:
        if (d - prev).days > gap_days:
            eps.append((start, prev)); start = d
        prev = d
    eps.append((start, prev))
    return eps


def main():
    df, panel = lab6_frame()
    sd = SeqData(df, "y_logrv_5", lookback=WIN, horizon=0)
    ae = LSTMAutoencoder(len(INPUTS), hidden=32, latent=8)
    ae, hist = train(ae, sd["train"].X, None, sd["val"].X, None, loss="recon", lr=1e-3, batch=64,
                     max_epochs=150, patience=15, seed=C.SEED)
    torch.save(ae.state_dict(), C.RESULTS_DIR / "lab6_ae.pt")
    err = {k: recon_error(ae, sd[k].X) for k in ("train", "val", "test")}
    thr = float(np.quantile(err["val"], 0.99))
    dte = sd["test"].dates
    flag = err["test"] > thr
    num = {"win": WIN, "thr": thr, "ae_epochs": len(hist), "n_flag": int(flag.sum()),
           "share_flag": 100 * flag.mean(), "err_train_med": float(np.median(err["train"])),
           "err_val_med": float(np.median(err["val"])), "err_test_med": float(np.median(err["test"]))}

    # per-channel contribution for each test window
    ae.eval()
    with torch.no_grad():
        Xt = torch.as_tensor(sd["test"].X)
        ch = ((ae(Xt) - Xt) ** 2).mean(dim=1).numpy()          # (n, channels)
    eps = episodes(dte[flag])
    rows = []
    for a, b in eps:
        m = (dte >= a) & (dte <= b) & flag
        k = np.argmax(np.where(m, err["test"], -1))
        top = np.argsort(ch[k])[::-1][:2]
        seg = panel.loc[a - pd.Timedelta(days=30):b, "adj_close"]
        dd = float(seg.min() / seg.loc[:a].max() - 1) if len(seg.loc[:a]) else np.nan
        rows.append((a.strftime("%Y-%m-%d"), b.strftime("%Y-%m-%d"), int(m.sum()), float(err["test"][k] / thr),
                     f"{INPUTS[top[0]]}, {INPUTS[top[1]]}".replace("_", "\\_"),
                     float(panel.loc[a:b, "VIXCLS"].max()), 100 * dd))
    ep = pd.DataFrame(rows, columns=["Start", "End", "Days", "Max error / threshold", "Main channels",
                                     "Max VIX", "Drawdown, %"]).set_index("Start")
    save_table("lab6_anomaly_episodes", ep, nd={"Days": 0, "Max error / threshold": 2, "Max VIX": 1, "Drawdown, %": 1},
               col_format="lcccp{3.6cm}cc")
    print(ep.to_string())
    num["n_episodes"] = len(eps)

    # baseline: |r| > 4 * rolling 1-year sd (computed with past data only)
    r = df["ret"]
    z = (r / r.rolling(252).std().shift(1)).abs()
    base_days = z.loc[dte].index[z.loc[dte] > 4]
    near = lambda d, S: np.any(np.abs((S - d).days) <= 5) if len(S) else False
    ae_days = dte[flag]
    num.update({"n_base": len(base_days),
                "base_caught": int(sum(near(d, ae_days) for d in base_days)),
                "ae_explained": 100 * np.mean([near(d, base_days) for d in ae_days]) if len(ae_days) else 0.0})

    # ---------------------------------------------------------------- figure
    f, axs = P.fig(4.4, nrows=2, sharex=True)
    px = panel.loc[dte, "adj_close"]
    axs[0].plot(dte, px, color=P.BLUE, lw=0.9)
    axs[0].scatter(dte[flag], px[flag], s=10, color=P.RED, zorder=3, label="AE anomaly")
    axs[0].scatter(base_days, panel.loc[base_days, "adj_close"], s=26, facecolor="none", edgecolor=P.INK,
                   lw=0.8, zorder=4, label="$|r_t| > 4\\sigma$")
    axs[0].set_ylabel("S&P 500"); axs[0].legend(loc="upper left", fontsize=8)
    axs[1].plot(dte, err["test"], color=P.BLUE, lw=0.7)
    axs[1].axhline(thr, color=P.RED, lw=0.9, ls="--", label="threshold (99% of validation)")
    axs[1].set_yscale("log"); axs[1].set_ylabel("reconstruction MSE"); axs[1].legend(loc="upper right", fontsize=8)
    f.tight_layout(); P.save(f, "lab6_anomaly")

    f, ax = P.fig(2.4)
    bins = np.logspace(np.log10(min(e.min() for e in err.values())), np.log10(max(e.max() for e in err.values())), 60)
    for k, col in [("train", P.BLUE), ("val", P.AQUA), ("test", P.ORANGE)]:
        ax.hist(err[k], bins=bins, density=True, histtype="step", lw=1.2, color=col, label=k)
    ax.axvline(thr, color=P.RED, ls="--", lw=0.9)
    ax.set_xscale("log"); ax.set_xlabel("reconstruction MSE of a 20-day window"); ax.set_ylabel("density")
    ax.legend(fontsize=8)
    P.save(f, "lab6_anomaly_hist")
    save_numbers("lab6n", num)
    print(num)


if __name__ == "__main__":
    main()
