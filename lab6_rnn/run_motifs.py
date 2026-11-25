"""Lab 6.4. Motif discovery in the S&P 500 by clustering subsequences.

Part A reproduces the warning of Keogh & Lin (2005): k-means applied to
*all* sliding windows of a series returns smooth, sine-like centroids that
are almost the same for a random walk - the result is driven by the
overlap of windows, not by the data.

Part B: meaningful motif discovery.
  1. z-normalised 30-day windows of log price taken with stride 30
     (no trivial matches) from the training block;
  2. embedding by an LSTM autoencoder (shape-only, univariate);
  3. k-means on the embeddings; medoids are reported as motifs;
  4. out-of-sample check on the test block: do motifs differ in the
     returns / volatility that follow them?

Run:  python -m lab6_rnn.run_motifs
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from scipy import stats
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from dmfin import config as C
from dmfin import plotting as P
from dmfin.reporting import save_numbers, save_table

from .common import lab6_frame
from .nets import LSTMAutoencoder, predict, set_seed, train

L = 30
torch.set_num_threads(16)


def znorm_windows(x: np.ndarray, starts: np.ndarray) -> np.ndarray:
    W = np.stack([x[s:s + L] for s in starts])
    mu, sd = W.mean(1, keepdims=True), W.std(1, keepdims=True) + 1e-8
    return (W - mu) / sd


def main():
    df, panel = lab6_frame()
    logp = np.log(panel["adj_close"]).reindex(df.index).to_numpy()
    idx = df.index
    num = {"L": L}

    # ------------------------------------------------------------- Part A
    rng = np.random.default_rng(C.SEED)
    rw = np.cumsum(rng.normal(0, 0.012, len(logp)))
    cents = {}
    for name, x in (("S&P 500", logp), ("random walk", rw)):
        W = znorm_windows(x, np.arange(0, len(x) - L))
        km = KMeans(3, n_init=20, random_state=C.SEED).fit(W)
        cents[name] = km.cluster_centers_
    A, B = cents["S&P 500"], cents["random walk"]
    cost = -np.corrcoef(A, B)[:3, 3:]
    r_, c_ = linear_sum_assignment(cost)
    num["keogh_corr"] = float(np.mean(-cost[r_, c_]))
    f, axs = P.fig(2.4, ncols=2, sharey=True)
    for ax, (name, Cc) in zip(axs, [("S&P 500", A), ("random walk", B[c_])]):
        for j in range(3):
            ax.plot(np.arange(L), Cc[j], color=P.CAT[j], lw=1.3)
        ax.set_title(f"{name}: k-means on all sliding windows", fontsize=9); ax.set_xlabel("day in window")
    axs[0].set_ylabel("z-normalized level")
    f.tight_layout(); P.save(f, "lab6_keogh")

    # ------------------------------------------------------------- Part B
    tr_end = np.searchsorted(idx, pd.Timestamp(C.TRAIN_END), side="right")
    va_end = np.searchsorted(idx, pd.Timestamp(C.VAL_END), side="right")
    st_tr = np.arange(0, tr_end - L - 5, L)                         # non-overlapping
    st_va = np.arange(tr_end, va_end - L - 5, 5)
    st_te = np.arange(va_end, len(idx) - L - 5, 5)                  # stride 5 for testing
    Wtr, Wva, Wte = (znorm_windows(logp, s)[..., None].astype(np.float32) for s in (st_tr, st_va, st_te))
    # augment the small training set with shifted (stride-5) windows for the AE only
    st_aug = np.arange(0, tr_end - L - 5, 5)
    Waug = znorm_windows(logp, st_aug)[..., None].astype(np.float32)
    set_seed(C.SEED)
    ae = LSTMAutoencoder(1, hidden=32, latent=4)
    ae, hist = train(ae, Waug, None, Wva, None, loss="recon", lr=2e-3, batch=64, max_epochs=200,
                     patience=20, seed=C.SEED)
    enc = lambda W: ae.encode(torch.as_tensor(W)).detach().numpy()
    Etr, Ete = enc(Wtr), enc(Wte)
    mu, sd = Etr.mean(0), Etr.std(0)
    Etr, Ete = (Etr - mu) / sd, (Ete - mu) / sd
    sil = {k: silhouette_score(Etr, KMeans(k, n_init=50, random_state=C.SEED).fit(Etr).labels_) for k in range(3, 9)}
    K = max(sil, key=sil.get)
    km = KMeans(K, n_init=100, random_state=C.SEED).fit(Etr)
    num.update({"n_train_windows": len(st_tr), "n_test_windows": len(st_te), "K": K, "sil": sil[K],
                "ae_epochs": len(hist)})

    # random-walk control: same pipeline on a random walk
    Wrw = znorm_windows(rw, st_tr)[..., None].astype(np.float32)
    Erw = (enc(Wrw) - mu) / sd
    num["sil_rw"] = float(silhouette_score(Erw, KMeans(K, n_init=50, random_state=C.SEED).fit(Erw).labels_))

    # forward behaviour after each motif, out of sample (test block)
    ret = df["ret"].to_numpy()
    fwd_ret = np.array([ret[s + L:s + L + 5].sum() for s in st_te])
    fwd_vol = np.array([np.sqrt(252 / 5 * (ret[s + L:s + L + 5] ** 2).sum()) for s in st_te])
    lab_te = km.predict(Ete)
    lab_tr = km.labels_
    rows = []
    medoids = []
    for c in range(K):
        mem = np.where(lab_tr == c)[0]
        dist = ((Etr[mem] - km.cluster_centers_[c]) ** 2).sum(1)
        medoids.append(mem[np.argmin(dist)])
        sel = lab_te == c
        rows.append((f"M{c+1}", len(mem), int(sel.sum()), 100 * Wtr[mem, -1, 0].mean() / 100,
                     100 * fwd_ret[sel].mean() if sel.any() else np.nan,
                     100 * fwd_vol[sel].mean() if sel.any() else np.nan))
    mt = pd.DataFrame(rows, columns=["Motif", "Train windows", "Test windows", "Mean end level (z)",
                                     "Next-5-day return, %", "Next-5-day vol., %"]).set_index("Motif")
    save_table("lab6_motifs", mt, nd={"Train windows": 0, "Test windows": 0, "Mean end level (z)": 2,
                                      "Next-5-day return, %": 2, "Next-5-day vol., %": 1})
    print(mt.to_string())
    groups_r = [fwd_ret[lab_te == c] for c in range(K) if (lab_te == c).sum() > 3]
    groups_v = [fwd_vol[lab_te == c] for c in range(K) if (lab_te == c).sum() > 3]
    num.update({"kw_ret_p": float(stats.kruskal(*groups_r).pvalue), "kw_vol_p": float(stats.kruskal(*groups_v).pvalue),
                "uncond_vol": 100 * fwd_vol.mean(), "uncond_ret": 100 * fwd_ret.mean()})

    f, axs = P.fig(2.2 * ((K + 3) // 4), nrows=(K + 3) // 4, ncols=4, sharex=True, sharey=True)
    axs = np.atleast_2d(axs).ravel()
    t = np.arange(L)
    for c in range(K):
        ax = axs[c]
        mem = Wtr[lab_tr == c, :, 0]
        q1, q2, q3 = np.quantile(mem, [0.25, 0.5, 0.75], axis=0)
        ax.fill_between(t, q1, q3, color=P.CAT[c], alpha=0.25, lw=0)
        ax.plot(t, q2, color=P.CAT[c], lw=1.0, ls="--")
        ax.plot(t, Wtr[medoids[c], :, 0], color=P.INK, lw=1.0)
        ax.set_title(f"M{c+1} (n={len(mem)})", fontsize=8.5)
    for ax in axs[K:]:
        ax.set_visible(False)
    for ax in axs[-4:]:
        ax.set_xlabel("day")
    f.tight_layout(); P.save(f, "lab6_motifs")
    save_numbers("lab6m", num)
    print(num, sil)


if __name__ == "__main__":
    main()
