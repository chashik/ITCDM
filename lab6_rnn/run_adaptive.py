"""Lab 6.5. LSTM + error-predicting network: correction and adaptive learning.

Idea (hybrid residual modelling, Zhang 2003; stacked generalisation,
Wolpert 1992): a second network g learns to predict the error of the base
forecaster f from information available at time t, and is used

  1. for correction:          y_hat = f(x_t) + lambda * g(z_t);
  2. for adaptation control:  during walk-forward fine-tuning of f the
     learning rate is scaled by the error level signalled by g and by the
     recently realised errors.

The error network must be trained on *out-of-sample* residuals of f.
In-sample residuals are too small and too regular; residuals of a single
validation block are too few.  We therefore produce out-of-fold residuals
for 2005-2014 by expanding-window refits of f (as in stacking) and use the
validation block only for early stopping and for the shrinkage lambda.

Timing: a 5-day target dated t is realised at t+5, so at time t only
e_{t-5}, e_{t-6}, ... are known, and fine-tuning may use labels <= t-5.

Run:  python -m lab6_rnn.run_adaptive
"""
from __future__ import annotations

import copy
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LinearRegression
from torch import nn

from dmfin import config as C
from dmfin import plotting as P
from dmfin.reporting import save_numbers, save_table
from dmfin.splits import make_windows

from .common import lab6_frame, rmse
from .nets import INPUTS, SeqData, SeqNet, diebold_mariano, predict, set_seed, train

LOOKBACK, H, HIDDEN = 60, 5, 32
N_LAGS, STEP = 10, 21
HAR = ["har_pk_d", "har_pk_w", "har_pk_m", "log_vix"]
CTX = ["log_vix", "d_log_vix", "log_pk_d", "ret", "term_10y3m", "credit_baa_aaa"]
OOF_CUTS = ["2005-01-01", "2007-01-01", "2009-01-01", "2011-01-01", "2013-01-01", "2015-01-01"]
BASES = {"returns": ["ret"], "full": INPUTS}
SEEDS = [0, 1, 2, 3, 4]


class ErrorNet(nn.Module):
    def __init__(self, n_in):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(n_in, 32), nn.ReLU(), nn.Dropout(0.1),
                               nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 1))

    def forward(self, x):
        return self.f(x).squeeze(-1)


def err_features(pos, e, fcast, Vctx, ctx_inputs):
    """Inputs of the error network at label positions ``pos`` (all known at t)."""
    ci = [ctx_inputs.index(c) for c in CTX]
    rows = []
    for t in np.atleast_1d(pos):
        lags = [e[t - H - k] for k in range(N_LAGS)]
        recent = np.nanmean(np.abs(e[t - H - 21:t - H + 1]))
        rows.append(lags + [fcast[t], recent] + list(Vctx[t, ci]))
    return np.nan_to_num(np.array(rows, dtype=np.float32))


def fit_lstm(X, y, Xv, yv, n_in, seed):
    set_seed(seed)                     # initial weights depend on the seed only
    m = SeqNet(n_in, HIDDEN, 1, "lstm")
    m, _ = train(m, X, y, Xv, yv, lr=1e-3, batch=64, max_epochs=100, patience=10, weight_decay=1e-5, seed=seed)
    return m


def base_forecasts(df, inputs, seed):
    """Out-of-fold forecasts on 2005-2014 + forecasts of the final model afterwards."""
    sd = SeqData(df, "y_logrv_5", lookback=LOOKBACK, horizon=H, inputs=inputs)
    V, yall, idx = sd.V, sd.yall, df.index
    trp = sd["train"].pos
    f = np.full(len(idx), np.nan)
    cuts = [pd.Timestamp(c) for c in OOF_CUTS]
    for a, b in zip(cuts[:-1], cuts[1:]):
        fitp = trp[idx[trp] < a - pd.Timedelta(days=10)]          # purge 5+ trading days
        X, y, _ = make_windows(V, yall, fitp, LOOKBACK)
        c = int(0.85 * len(y))
        m = fit_lstm(X[:c], y[:c], X[c:], y[c:], len(inputs), seed)
        pp = trp[(idx[trp] >= a) & (idx[trp] < b)]
        Xp, _, pp = make_windows(V, yall, pp, LOOKBACK)
        f[pp] = predict(m, Xp)
    final = fit_lstm(sd["train"].X, sd["train"].y, sd["val"].X, sd["val"].y, len(inputs), seed)
    rest = np.arange(sd["val"].pos.min(), len(idx))
    X, _, rest = make_windows(V, yall, rest, LOOKBACK)
    f[rest] = predict(final, X)
    known = ~np.isnan(yall); known[len(idx) - H:] = False
    e = np.where(known & ~np.isnan(f), yall - f, np.nan)
    return sd, final, f, e


def fit_error_net(Ztr, etr, Zva, eva, seed):
    mu, sd = Ztr.mean(0), Ztr.std(0) + 1e-6
    S = lambda Z: ((Z - mu) / sd).astype(np.float32)
    set_seed(seed)
    g = ErrorNet(Ztr.shape[1])
    g, _ = train(g, S(Ztr), etr.astype(np.float32), S(Zva), eva.astype(np.float32), lr=1e-3, batch=32,
                 max_epochs=300, patience=25, weight_decay=1e-3, seed=seed)
    gv = predict(g, S(Zva))
    lams = np.linspace(0, 1, 21)
    lam = float(lams[np.argmin([np.mean((eva - l * gv) ** 2) for l in lams])])
    return (lambda Z: predict(g, S(Z))), lam


def run_seed(seed):
    torch.set_num_threads(2)
    df, _ = lab6_frame()
    ctx = SeqData(df, "y_logrv_5", lookback=LOOKBACK, horizon=H)       # context channels
    out, info = {}, {}
    for bname, inputs in BASES.items():
        sd, final, f, e = base_forecasts(df, inputs, seed)
        idx = df.index
        va, te, trp = sd["val"].pos, sd["test"].pos, sd["train"].pos
        oofp = trp[idx[trp] >= pd.Timestamp("2005-03-01")]
        Z = lambda pos: err_features(pos, e, f, ctx.V, ctx.inputs)
        # (a) naive: error net trained on validation residuals only
        cut = int(0.75 * len(va))
        g_naive, lam_naive = fit_error_net(Z(va[:cut]), e[va[:cut]], Z(va[cut:]), e[va[cut:]], seed)
        # (b) stacked: out-of-fold residuals 2005-2014, early stopping on validation
        g, lam = fit_error_net(Z(oofp), e[oofp], Z(va), e[va], seed)
        gt = g(Z(te))
        out[f"{bname}_static"] = f[te]
        out[f"{bname}_naive"] = f[te] + lam_naive * g_naive(Z(te))
        out[f"{bname}_stacked"] = f[te] + lam * gt
        info[bname] = {"lam": lam, "lam_naive": lam_naive, "corr": float(np.corrcoef(gt, e[te])[0, 1]),
                       "oof_rmse": float(np.sqrt(np.nanmean(e[oofp] ** 2))), "n_oof": len(oofp),
                       "ghat": gt, "e": e[te]}
        if bname != "full":
            continue
        # ------------------------------------------ walk-forward fine-tuning (full model)
        V, yall = sd.V, sd.yall
        val_mae = float(np.nanmean(np.abs(e[va])))

        def walk(mode, lr0, epochs, window, correct):
            model = copy.deepcopy(final)
            fl, el = f.copy(), e.copy()
            el[te.min():] = np.nan
            preds, mults = np.full(len(te), np.nan), []
            for k0 in range(0, len(te), STEP):
                s = te[k0]
                if mode == "error":
                    zs = err_features([s], el, fl, ctx.V, ctx.inputs)
                    recent = float(np.nanmean(np.abs(el[s - H - 21:s - H + 1])))
                    mult = float(np.clip((0.5 * recent + 0.5 * abs(g(zs)[0])) / val_mae, 0.25, 5.0) ** 2)
                else:
                    mult = 1.0
                mults.append((idx[s], mult))
                lab_pos = np.arange(LOOKBACK - 1, s - H + 1)
                if window:
                    lab_pos = lab_pos[-window:]
                Xw, yw, _ = make_windows(V, yall, lab_pos, LOOKBACK)
                opt = torch.optim.Adam(model.parameters(), lr=lr0 * mult)
                set_seed(seed * 1000 + k0); model.train()
                Xt, yt = torch.as_tensor(Xw), torch.as_tensor(yw)
                for _ in range(epochs):
                    perm = torch.randperm(len(Xt))
                    for i in range(0, len(perm), 64):
                        b = perm[i:i + 64]
                        opt.zero_grad()
                        nn.functional.mse_loss(model(Xt[b]), yt[b]).backward()
                        nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
                seg = te[k0:k0 + STEP]
                Xs, _, _ = make_windows(V, yall, seg, LOOKBACK)
                ps = predict(model, Xs); fl[seg] = ps
                if correct:
                    ps = ps + lam * g(err_features(seg, el, fl, ctx.V, ctx.inputs))
                preds[k0:k0 + len(seg)] = ps
                el[seg] = yall[seg] - fl[seg]       # becomes usable H days later (lags >= H)
            return preds, mults

        out["ft_recent_fixed"], _ = walk("fixed", 2e-4, 2, 1000, False)
        out["ft_replay_fixed"], _ = walk("fixed", 5e-5, 1, None, False)
        out["ft_replay_error"], mults = walk("error", 5e-5, 1, None, False)
        out["ft_replay_error_corr"], _ = walk("error", 5e-5, 1, None, True)
        info["mults"] = mults
    return seed, out, info


def main():
    df, _ = lab6_frame()
    sd = SeqData(df, "y_logrv_5", lookback=LOOKBACK, horizon=H)
    te, dte, yte = sd["test"].pos, sd["test"].dates, sd["test"].y
    with ProcessPoolExecutor(len(SEEDS)) as ex:
        res = list(ex.map(run_seed, SEEDS))

    tr_end = sd["val"].pos.max()
    har_static = LinearRegression().fit(df.iloc[:tr_end + 1][HAR], df.iloc[:tr_end + 1]["y_logrv_5"]).predict(df.loc[dte, HAR])
    har_roll = np.empty(len(te))
    for k0 in range(0, len(te), STEP):
        fit = df.iloc[:te[k0] - H + 1]
        seg = slice(k0, k0 + STEP)
        har_roll[seg] = LinearRegression().fit(fit[HAR], fit["y_logrv_5"]).predict(df.iloc[te[seg]][HAR])

    names = {"returns_static": "Returns-only LSTM", "returns_naive": "Returns-only + error net (valid. resid.)",
             "returns_stacked": "Returns-only + error net (out-of-fold resid.)",
             "full_static": "Full LSTM, static", "full_naive": "Full LSTM + error net (valid. resid.)",
             "full_stacked": "Full LSTM + error net (out-of-fold resid.)",
             "ft_recent_fixed": "Full LSTM, fine-tuned on last 1000 windows",
             "ft_replay_fixed": "Full LSTM, fine-tuned with replay, fixed rate",
             "ft_replay_error": "Full LSTM, replay, error-driven rate",
             "ft_replay_error_corr": "Full LSTM, replay, error-driven rate + correction"}
    avg = {k: np.mean([r[1][k] for r in res], axis=0) for k in names}
    rows = [("HAR + VIX, static", np.nan, np.nan, rmse(yte, har_static), np.nan, np.nan),
            ("HAR + VIX, re-estimated every 21 days", np.nan, np.nan, rmse(yte, har_roll), np.nan, np.nan)]
    for k, name in names.items():
        per = [rmse(yte, r[1][k]) for r in res]
        ref = "returns_static" if k.startswith("returns") else "full_static"
        dm, p = diebold_mariano(yte - avg[k], yte - avg[ref], h=H) if k != ref else (np.nan, np.nan)
        rows.append((name, np.mean(per), np.std(per), rmse(yte, avg[k]), dm, p))
    tab = pd.DataFrame(rows, columns=["Model", "RMSE (mean of 5 seeds)", "SD", "RMSE of seed average",
                                      "DM", "$p$"]).set_index("Model")
    save_table("lab6_adaptive", tab, nd=3, col_format="p{4.8cm}ccccc")
    print(tab.round(4).to_string())

    num = {"har_roll": rmse(yte, har_roll), "har_static": rmse(yte, har_static), "step": STEP,
           "n_lags": N_LAGS, "n_oof": res[0][2]["full"]["n_oof"]}
    for k in names:
        num[f"rmse_{k}"] = float(tab.loc[names[k], "RMSE (mean of 5 seeds)"])
        num[f"sd_{k}"] = float(tab.loc[names[k], "SD"])
        num[f"avg_{k}"] = float(tab.loc[names[k], "RMSE of seed average"])
        num[f"p_{k}"] = float(tab.loc[names[k], "$p$"])
    for b in BASES:
        num[f"lam_{b}"] = float(np.mean([r[2][b]["lam"] for r in res]))
        num[f"lamnaive_{b}"] = float(np.mean([r[2][b]["lam_naive"] for r in res]))
        num[f"corr_{b}"] = float(np.mean([r[2][b]["corr"] for r in res]))
        num[f"oofrmse_{b}"] = float(np.mean([r[2][b]["oof_rmse"] for r in res]))
    num["gain_returns"] = 100 * (1 - num["rmse_returns_stacked"] / num["rmse_returns_static"])
    num["gain_full"] = 100 * (1 - num["rmse_full_stacked"] / num["rmse_full_static"])

    # ----------------------------------------- cumulative squared-error gains
    f, axs = P.fig(4.6, nrows=2, sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    for k, col, ref in [("returns_stacked", P.AQUA, "returns_static"), ("full_stacked", P.BLUE, "full_static"),
                        ("ft_recent_fixed", P.RED, "full_static"), ("ft_replay_fixed", P.ORANGE, "full_static"),
                        ("ft_replay_error_corr", P.VIOLET, "full_static")]:
        lab = {"returns_stacked": "returns-only LSTM + error net", "full_stacked": "full LSTM + error net",
               "ft_recent_fixed": "fine-tuned on recent windows", "ft_replay_fixed": "fine-tuned with replay",
               "ft_replay_error_corr": "replay + error-driven rate + correction"}[k]
        axs[0].plot(dte, np.cumsum((yte - avg[ref]) ** 2 - (yte - avg[k]) ** 2), color=col, lw=1.1, label=lab)
    axs[0].axhline(0, color=P.INK2, lw=0.6)
    axs[0].set_ylabel("cumulative SSE reduction\nvs own static base"); axs[0].legend(fontsize=7, loc="upper left")
    m = pd.DataFrame(res[0][2]["mults"], columns=["date", "mult"]).set_index("date")["mult"]
    axs[1].step(m.index, m.values, where="post", color=P.BLUE, lw=1.0)
    axs[1].set_yscale("log"); axs[1].set_ylabel("learning-rate\nmultiplier")
    f.tight_layout(); P.save(f, "lab6_adaptive")
    num.update({"mult_max": float(m.max()), "mult_max_date": m.idxmax().strftime("%Y-%m-%d"),
                "mult_median": float(m.median()),
                "corr_mult_vix": float(np.corrcoef(m.values, np.exp(df.loc[m.index, "log_vix"]))[0, 1])})

    # -------------------------------------------- predicted vs realised errors
    f, axs = P.fig(2.8, ncols=2, sharey=False)
    for ax, b, ttl in [(axs[0], "returns", "returns-only LSTM"), (axs[1], "full", "full LSTM")]:
        gh, ee = res[0][2][b]["ghat"], res[0][2][b]["e"]
        ax.scatter(gh, ee, s=4, color=P.BLUE, alpha=0.45, lw=0)
        b1, b0 = np.polyfit(gh, ee, 1)
        xs = np.linspace(gh.min(), gh.max(), 10)
        ax.plot(xs, b0 + b1 * xs, color=P.ORANGE, lw=1.0)
        ax.set_title(f"{ttl}: corr = {np.corrcoef(gh, ee)[0, 1]:.2f}", fontsize=9)
        ax.set_xlabel("predicted error $\\hat e_t$"); ax.set_ylabel("realized error $e_t$")
    f.tight_layout(); P.save(f, "lab6_error_scatter")
    save_numbers("lab6a", num)
    print(num)


if __name__ == "__main__":
    main()
