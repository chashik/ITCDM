"""Lab 6.1. Forecasting next-week realised volatility with RNN, LSTM and GRU.

Run:  python -m lab6_rnn.run_forecast
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LinearRegression

from dmfin import config as C
from dmfin import plotting as P
from dmfin.reporting import save_numbers, save_table

from .common import lab6_frame, r2, rmse
from .nets import INPUTS, SeqData, SeqNet, diebold_mariano, predict, set_seed, train

LOOKBACK, HORIZON, HIDDEN = 60, 5, 32
CELLS = {"rnn": "Elman RNN", "lstm": "LSTM", "gru": "GRU"}
SEEDS = [0, 1, 2, 3, 4]
HAR = ["har_pk_d", "har_pk_w", "har_pk_m", "log_vix"]


def _job(args):
    cell, seed = args
    torch.set_num_threads(2)
    df, _ = lab6_frame()
    sd = SeqData(df, "y_logrv_5", lookback=LOOKBACK, horizon=HORIZON)
    set_seed(seed)                     # initial weights depend on the seed only
    m = SeqNet(len(INPUTS), HIDDEN, 1, cell)
    m, hist = train(m, sd["train"].X, sd["train"].y, sd["val"].X, sd["val"].y, lr=1e-3, batch=64,
                    max_epochs=100, patience=10, weight_decay=1e-5, seed=seed)
    if cell == "lstm" and seed == 0:
        torch.save(m.state_dict(), C.RESULTS_DIR / "lab6_lstm_base.pt")
    n_par = sum(p.numel() for p in m.parameters())
    return cell, seed, predict(m, sd["val"].X), predict(m, sd["test"].X), hist, n_par


def main():
    df, panel = lab6_frame()
    sd = SeqData(df, "y_logrv_5", lookback=LOOKBACK, horizon=HORIZON)
    yva, yte = sd["val"].y, sd["test"].y
    dva, dte = sd["val"].dates, sd["test"].dates
    num = {"n_train": len(sd["train"].y), "n_val": len(yva), "n_test": len(yte), "lookback": LOOKBACK,
           "hidden": HIDDEN, "first_window": sd["train"].dates.min().strftime("%Y-%m-%d"),
           "n_inputs": len(INPUTS)}

    with ProcessPoolExecutor(15) as ex:
        out = list(ex.map(_job, [(c, s) for c in CELLS for s in SEEDS]))
    preds = {c: {"val": [], "test": [], "hist": [], "npar": 0} for c in CELLS}
    for cell, seed, pv, pt, hist, n_par in out:
        preds[cell]["val"].append(pv); preds[cell]["test"].append(pt)
        preds[cell]["hist"].append(hist); preds[cell]["npar"] = n_par

    tr = df.loc[sd["train"].dates]; trva = pd.concat([tr, df.loc[dva]])
    har = LinearRegression().fit(tr[HAR], tr["y_logrv_5"])
    har_va = har.predict(df.loc[dva, HAR])
    har_te = LinearRegression().fit(trva[HAR], trva["y_logrv_5"]).predict(df.loc[dte, HAR])
    naive_va, naive_te = df.loc[dva, "log_rv_5_past"].to_numpy(), df.loc[dte, "log_rv_5_past"].to_numpy()

    rows = [("Naive (last 5-day RV)", 0, rmse(yva, naive_va), np.nan, rmse(yte, naive_te), np.nan, r2(yte, naive_te)),
            ("HAR + VIX, OLS", 5, rmse(yva, har_va), np.nan, rmse(yte, har_te), np.nan, r2(yte, har_te))]
    ens = {}
    for c, name in CELLS.items():
        v = [rmse(yva, p) for p in preds[c]["val"]]; t = [rmse(yte, p) for p in preds[c]["test"]]
        ens[c] = (np.mean(preds[c]["val"], axis=0), np.mean(preds[c]["test"], axis=0))
        rows.append((f"{name}, single seed (mean)", preds[c]["npar"], np.mean(v), np.std(v), np.mean(t), np.std(t),
                     np.mean([r2(yte, p) for p in preds[c]["test"]])))
        rows.append((f"{name}, 5-seed ensemble", preds[c]["npar"] * 5, rmse(yva, ens[c][0]), np.nan,
                     rmse(yte, ens[c][1]), np.nan, r2(yte, ens[c][1])))
    tab = pd.DataFrame(rows, columns=["Model", "Parameters", "Valid. RMSE", "SD", "Test RMSE", "SD ", "Test $R^2$"]).set_index("Model")
    save_table("lab6_forecast", tab, nd={"Parameters": 0, "Valid. RMSE": 3, "SD": 3, "Test RMSE": 3, "SD ": 3, "Test $R^2$": 3},
               col_format="p{5.0cm}ccccccc")
    print(tab.round(4).to_string())
    for c in CELLS:
        s, p = diebold_mariano(yte - ens[c][1], yte - har_te, h=HORIZON)
        num[f"dm_{c}_stat"], num[f"dm_{c}_p"] = s, p
    best = min(CELLS, key=lambda c: rmse(yva, ens[c][0]))
    num.update({"best_cell": CELLS[best], "har_test": rmse(yte, har_te), "naive_test": rmse(yte, naive_te),
                "har_val": rmse(yva, har_va), "har_r2": r2(yte, har_te),
                **{f"{c}_ens_test": rmse(yte, ens[c][1]) for c in CELLS},
                **{f"{c}_ens_val": rmse(yva, ens[c][0]) for c in CELLS},
                **{f"{c}_seed_sd": float(np.std([rmse(yte, p) for p in preds[c]["test"]])) for c in CELLS},
                **{f"{c}_npar": preds[c]["npar"] for c in CELLS},
                **{f"{c}_epochs": float(np.mean([len(h) for h in preds[c]["hist"]])) for c in CELLS},
                "lstm_ens_r2": r2(yte, ens["lstm"][1])})
    pd.DataFrame({"y": yte, "har": har_te, "naive": naive_te, **{c: ens[c][1] for c in CELLS}},
                 index=dte).to_csv(C.RESULTS_DIR / "lab6_forecast_test.csv")
    pd.DataFrame({"y": yva, "har": har_va, **{c: ens[c][0] for c in CELLS}},
                 index=dva).to_csv(C.RESULTS_DIR / "lab6_forecast_val.csv")

    # ------------------------------------------------------- learning curves
    f, axs = P.fig(2.6, ncols=3, sharey=True)
    for ax, c in zip(axs, CELLS):
        for k, h in enumerate(preds[c]["hist"]):
            ax.plot(np.arange(1, len(h) + 1), h[:, 0], color=P.BLUE, lw=0.8, alpha=0.8, label="training" if k == 0 else None)
            ax.plot(np.arange(1, len(h) + 1), h[:, 1], color=P.ORANGE, lw=0.8, alpha=0.8, label="validation" if k == 0 else None)
        ax.set_title(CELLS[c], fontsize=9); ax.set_xlabel("epoch")
    axs[0].set_ylabel("MSE of log RV"); axs[0].legend(fontsize=7.5); axs[0].set_ylim(0.1, 0.6)
    f.tight_layout(); P.save(f, "lab6_learning_curves")

    # ---------------------------------------------------- test-period forecast
    f, axs = P.fig(4.4, nrows=2)
    for ax, (a, b) in zip(axs, [(dte.min(), dte.max()), (pd.Timestamp("2020-01-15"), pd.Timestamp("2020-07-31"))]):
        m = (dte >= a) & (dte <= b)
        ax.plot(dte[m], 100 * np.exp(yte[m]), color=P.INK2, lw=0.8, label="realized, next 5 days")
        ax.plot(dte[m], 100 * np.exp(har_te[m]), color=P.ORANGE, lw=0.9, label="HAR + VIX")
        ax.plot(dte[m], 100 * np.exp(ens["lstm"][1][m]), color=P.BLUE, lw=1.0, label="LSTM ensemble")
        ax.set_yscale("log"); ax.set_ylabel("volatility, %")
        from matplotlib.ticker import FixedLocator, NullFormatter, ScalarFormatter
        ax.yaxis.set_major_locator(FixedLocator([5, 10, 20, 40, 80, 160]))
        ax.yaxis.set_major_formatter(ScalarFormatter()); ax.yaxis.set_minor_formatter(NullFormatter())
    axs[0].legend(loc="upper right", fontsize=8)
    f.tight_layout(); P.save(f, "lab6_forecast_series")

    # ------------------------------------------------------- error by year
    yrs = dte.year
    e = pd.DataFrame({"HAR + VIX": (yte - har_te) ** 2, "LSTM": (yte - ens["lstm"][1]) ** 2,
                      "GRU": (yte - ens["gru"][1]) ** 2}, index=dte).groupby(yrs).mean() ** 0.5
    f, ax = P.fig(2.5)
    w = 0.27
    for j, (col, cc) in enumerate(zip(e.columns, [P.ORANGE, P.BLUE, P.AQUA])):
        ax.bar(e.index + (j - 1) * w, e[col], width=w, color=cc, label=col)
    ax.set_xticks(e.index); ax.set_ylabel("RMSE of log RV"); ax.legend(fontsize=8, ncol=3)
    P.save(f, "lab6_error_by_year")
    num.update({"rmse_2020_lstm": float(e.loc[2020, "LSTM"]), "rmse_2020_har": float(e.loc[2020, "HAR + VIX"]),
                "years_lstm_better": int((e["LSTM"] < e["HAR + VIX"]).sum()), "n_years": len(e)})
    save_numbers("lab6f", num)
    print(num)


if __name__ == "__main__":
    main()
