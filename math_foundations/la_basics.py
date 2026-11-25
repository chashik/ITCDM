"""Part 1, chapter 2 (mathematical foundations), section "Vectors, matrices and tensors": the linear algebra that
all parts of the guide use, checked on their data.

1. Shapes: the data matrix of Lab 1 (stocks x features), the design matrix of Lab 2, the weight matrices of a Lab 3
   network and the minibatch tensor of the Lab 6 LSTM, with the number of multiplications of one forward pass.
2. Correlation is the cosine of the angle between centred vectors (Boyd and Vandenberghe, p. 60): all pairs of the Lab 1
   stocks.
3. Portfolio variance w^T S w equals the variance of the portfolio returns; diversification of the equal-weight portfolio
   of the Lab 1 stocks against the average correlation.
4. Eigenvalues of the correlation matrix of the Lab 1 stocks: none negative (positive semidefinite), their sum is the
   trace; the first eigenvector is the market (correlation of its scores with the S&P 500).
5. Normal equations and orthogonality on the regression of Lab 2: X^T (y - X beta) = 0, rank of X.
6. Gradients: 2/n X^T (X beta - y) against finite differences and automatic differentiation; the backward pass of a
   two-layer network written with matrix products against PyTorch's autograd on a minibatch of Lab 3.
Outputs: paper/generated/mathLin_numbers.tex, paper/figures/mathlin_pca.pdf.
Run (from code/):  python -m math_foundations.la_basics
"""
from __future__ import annotations

import matplotlib.ticker
import numpy as np
import pandas as pd
import torch

from dmfin import config as C
from dmfin import plotting as P
from dmfin.reporting import save_numbers, sci


def corr_as_cosine(R: np.ndarray) -> np.ndarray:
    """Matrix of cosines of the angles between the centred columns of R."""
    Z = R - R.mean(axis=0)
    Z = Z / np.linalg.norm(Z, axis=0)
    return Z.T @ Z


def backprop_two_layer(X, y, W1, b1, w2, b2):
    """Mean logistic loss of the network sigma(w2 . relu(W1 x + b1) + b2) on a minibatch and its gradients, computed
    with matrix products only (the chain rule backwards, layer by layer)."""
    Z = X @ W1.T + b1                          # B x K: pre-activations of the hidden layer
    H = np.maximum(Z, 0)                       # B x K
    s = H @ w2 + b2                            # B: logits
    p = 1 / (1 + np.exp(-s))
    loss = float(np.mean(np.logaddexp(0, s) - y * s))
    ds = (p - y) / len(y)                      # dL/ds
    dw2, db2 = H.T @ ds, ds.sum()              # K, scalar
    dZ = np.outer(ds, w2) * (Z > 0)            # B x K: back through w2 and the ReLU
    dW1, db1 = dZ.T @ X, dZ.sum(axis=0)        # K x p, K: the same shapes as W1 and b1
    return loss, {"W1": dW1, "b1": db1, "w2": dw2, "b2": db2}


def stocks():
    from dmfin.data import load_market_panel, load_stock_prices
    px = load_stock_prices()
    r = np.log(px).diff().iloc[1:]
    fit = r.loc[C.STOCKS_START:C.STOCKS_SPLIT].dropna(axis=1)
    hold = r.loc[pd.Timestamp(C.STOCKS_SPLIT) + pd.Timedelta(days=1):C.END, fit.columns].dropna()
    mkt = np.log(load_market_panel()["adj_close"]).diff()
    return fit, hold, mkt


def main():
    num = {}
    from lab1_clustering.model import FEATURES as L1
    from math_foundations.m1_linear_algebra import lab2_train
    fit, hold, mkt = stocks()
    n, p = fit.shape
    num.update({"lab1_stocks": p, "lab1_features": len(L1), "fit_days": n})

    # 1. shapes and the cost of one forward pass
    Xtr, ytr, _, _ = lab2_train()
    num.update({"lab2_rows": Xtr.shape[0], "lab2_cols": Xtr.shape[1]})
    from dmfin.data import load_credit
    from lab3_classification.model import prepare
    X3, y3 = prepare(load_credit())
    p3, width, batch = X3.shape[1], 256, 256
    num.update({"lab3_features": p3, "lab3_width": width, "lab3_batch": batch,
                "lab3_params": (p3 + 1) * width + (width + 1) * width + width + 1,
                "lab3_mults": batch * (p3 * width + width * width + width)})
    lstm = torch.nn.LSTM(8, 32, batch_first=True)
    num.update({"lab6_batch": 64, "lab6_lookback": 60, "lab6_inputs": 8, "lab6_hidden": 32,
                "lab6_lstm_params": sum(q.numel() for q in lstm.parameters()),
                "lab6_wih_rows": lstm.weight_ih_l0.shape[0]})
    out, _ = lstm(torch.zeros(64, 60, 8))
    num["lab6_out_shape"] = "\\times".join(str(s) for s in out.shape)

    # 2. correlation = cosine of centred vectors
    R = fit.to_numpy()
    num["cos_corr_maxdiff"] = sci(float(np.abs(corr_as_cosine(R) - np.corrcoef(R.T)).max()))

    # 3. portfolio variance of the equal-weight portfolio (hold-out, simple returns)
    Rh = np.expm1(hold.to_numpy())
    w = np.full(p, 1 / p)
    S = np.cov(Rh.T)
    var_p, var_direct = float(w @ S @ w), float(np.var(Rh @ w, ddof=1))
    sd = np.sqrt(np.diag(S))
    rho = (S / np.outer(sd, sd))[~np.eye(p, dtype=bool)].mean()
    approx = float(np.mean(sd ** 2) * (1 / p + (1 - 1 / p) * rho))
    num.update({"port_identity_diff": sci(abs(var_p - var_direct)), "port_vol": 100 * np.sqrt(252 * var_p),
                "stock_vol_mean": 100 * np.sqrt(252) * sd.mean(), "port_rho": rho,
                "port_vol_approx": 100 * np.sqrt(252 * approx), "hold_days": len(Rh)})

    # 4. eigenvalues of the correlation matrix of the fitting period
    Cm = np.corrcoef(R.T)
    lam, V = np.linalg.eigh(Cm)
    lam, V = lam[::-1], V[:, ::-1]
    v1 = V[:, 0] * np.sign(V[:, 0].sum())
    Zs = (R - R.mean(0)) / R.std(0)
    pc1 = Zs @ v1
    m = mkt.reindex(fit.index).to_numpy()
    ok = np.isfinite(m)
    num.update({"eig_min": float(lam.min()), "eig_sum": float(lam.sum()), "eig1": float(lam[0]),
                "eig1_share": 100 * lam[0] / p, "eig2": float(lam[1]), "eig_above_one": int((lam > 1).sum()),
                "v1_positive": int((v1 > 0).sum()), "pc1_mkt_corr": float(np.corrcoef(pc1[ok], m[ok])[0, 1]),
                "orth_err": sci(float(np.abs(V.T @ V - np.eye(p)).max()))})
    num["eig_top5_share"] = 100 * lam[:5].sum() / p

    # 5. normal equations and orthogonality (Lab 2, with an intercept)
    A = np.c_[np.ones(len(Xtr)), Xtr]
    beta, *_ = np.linalg.lstsq(A, ytr, rcond=None)
    beta_ne = np.linalg.solve(A.T @ A, A.T @ ytr)
    resid = ytr - A @ beta
    num.update({"ls_rank": int(np.linalg.matrix_rank(A)), "ls_cols": A.shape[1],
                "ls_orth": sci(float(np.abs(A.T @ resid).max() / np.abs(A.T @ ytr).max())),
                "ls_ne_diff": sci(float(np.abs(beta - beta_ne).max() / np.abs(beta).max()))})

    # 6. gradients: formula, finite differences, automatic differentiation
    rng = np.random.default_rng(C.SEED)
    b0 = rng.normal(size=A.shape[1])
    J = lambda b: float(np.mean((A @ b - ytr) ** 2))
    g = 2 / len(ytr) * A.T @ (A @ b0 - ytr)
    h = 1e-6
    gfd = np.array([(J(b0 + h * e) - J(b0 - h * e)) / (2 * h) for e in np.eye(len(b0))])
    bt = torch.tensor(b0, requires_grad=True)
    torch.mean((torch.tensor(A) @ bt - torch.tensor(ytr)) ** 2).backward()
    num.update({"grad_fd_err": sci(float(np.abs(g - gfd).max() / np.abs(g).max())),
                "grad_auto_err": sci(float(np.abs(g - bt.grad.numpy()).max() / np.abs(g).max()))})
    idx = rng.choice(len(X3), batch, replace=False)
    Xb = X3.to_numpy(float)[idx]
    Xb = (Xb - X3.to_numpy(float).mean(0)) / X3.to_numpy(float).std(0)
    yb = y3.to_numpy(float)[idx]
    K = 16
    W1, b1 = rng.normal(0, 0.3, (K, p3)), rng.normal(0, 0.1, K)
    w2, b2 = rng.normal(0, 0.3, K), 0.1
    loss, grads = backprop_two_layer(Xb, yb, W1, b1, w2, b2)
    T = {k: torch.tensor(v, requires_grad=True) for k, v in (("W1", W1), ("b1", b1), ("w2", w2), ("b2", np.array(b2)))}
    s = torch.relu(torch.tensor(Xb) @ T["W1"].T + T["b1"]) @ T["w2"] + T["b2"]
    lt = torch.nn.functional.binary_cross_entropy_with_logits(s, torch.tensor(yb))
    lt.backward()
    num.update({"bp_hidden": K, "bp_loss_diff": sci(abs(loss - float(lt.detach()))),
                "bp_grad_diff": sci(max(float(np.abs(grads[k] - T[k].grad.numpy()).max()) for k in grads))})

    # figure: two stocks with their principal axes; the spectrum of the correlation matrix
    c2 = np.corrcoef(R.T)
    np.fill_diagonal(c2, -1)
    i, j = np.unravel_index(np.argmax(c2), c2.shape)
    num.update({"pair_a": fit.columns[i], "pair_b": fit.columns[j], "pair_corr": float(c2[i, j])})
    f, axs = P.fig(3.0, ncols=2, gridspec_kw={"width_ratios": [1, 1.25]})
    xy = 100 * R[:, [i, j]]
    S2 = np.cov(xy.T)
    l2, V2 = np.linalg.eigh(S2)
    axs[0].scatter(xy[:, 0], xy[:, 1], s=2, color=P.MUTED, alpha=0.5, rasterized=True)
    for k, col, lab in ((1, P.BLUE, "first eigenvector"), (0, P.ORANGE, "second eigenvector")):
        d = 2 * np.sqrt(l2[k]) * V2[:, k]                      # +- two standard deviations along the axis
        axs[0].plot([-d[0], d[0]], [-d[1], d[1]], color=col, lw=2.0, label=lab)
    lim = np.percentile(np.abs(xy), 99.5)
    axs[0].set_xlim(-lim, lim); axs[0].set_ylim(-lim, lim); axs[0].set_aspect("equal")
    axs[0].set_xlabel(f"{fit.columns[i]}, daily log return, %"); axs[0].set_ylabel(f"{fit.columns[j]}, %")
    axs[0].legend(fontsize=7, loc="upper left", frameon=False)
    axs[1].bar(range(1, 31), lam[:30], color=[P.BLUE] + [P.MUTED] * 29)
    axs[1].axhline(1, color=P.INK, lw=0.7, ls="--")
    axs[1].set_yscale("log")
    axs[1].yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
    axs[1].set_xlabel("eigenvalue number (the largest 30 of %d)" % p)
    axs[1].set_ylabel("eigenvalue of the correlation matrix")
    axs[1].annotate(f"market: {100 * lam[0] / p:.0f} % of the variance", xy=(1.4, lam[0]), xytext=(6, lam[0] / 1.3),
                    fontsize=7.5, arrowprops=dict(arrowstyle="->", color=P.INK, lw=0.7))
    f.tight_layout()
    P.save(f, "mathlin_pca")
    save_numbers("mathLin", num)
    for k, v in num.items():
        print(f"{k:24s} {v}")


if __name__ == "__main__":
    main()
