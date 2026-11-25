"""Lab 6 (iteration 2). Global sequence models, a pretrained forecaster and conformal intervals.

* One network per architecture (LSTM, GRU, TCN, Transformer) is trained on all 62 series
  of the panel (``dmfin.panel``) and compared with a linear HAR model for each series, a
  pooled linear model and an LSTM trained on the S&P 500 alone;
* Chronos-Bolt and Chronos-2 (pretrained forecasters, zero-shot): each series is forecast
  from its own realized history; Chronos-2 also with log VIX as a covariate; the skill of
  each model before and after its release is compared;
* adaptive conformal prediction intervals for the S&P 500 forecasts of the global LSTM.

Run:  python -m lab6_rnn.run_global       (a GPU is used when available)
"""
from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LinearRegression

from dmfin import config as C
from dmfin import plotting as P
from dmfin.panel import FEATURES, build_panel, split_masks, window_index
from dmfin.reporting import save_numbers, save_table
from lab2_regression.run_ext import adaptive_conformal

from .global_models import n_params, predict_rows, train_global
from dmfin.device import large_work_device

from .nets import diebold_mariano

LOOKBACK, H = 20, 5
MODELS = {"lstm": "LSTM", "gru": "GRU", "tcn": "TCN", "transformer": "Transformer"}
SEEDS = [0, 1, 2]
HAR = ["lrv_d", "lrv_w", "lrv_m", "log_vix"]
CHRONOS = "amazon/chronos-bolt-small"
CHRONOS_RELEASE = "2024-11-25"                 # creation of the model repository on Hugging Face
CHRONOS2 = "amazon/chronos-2"
CHRONOS2_RELEASE = "2025-10-17"                # technical report arXiv:2510.15821 (repository created 2025-10-30)
PRETRAINED = {"Chronos-Bolt (zero-shot)": ("chronos", CHRONOS_RELEASE),
              "Chronos-2 (zero-shot)": ("chronos2", CHRONOS2_RELEASE),
              "Chronos-2 with VIX (zero-shot)": ("chronos2v", CHRONOS2_RELEASE)}


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def prepare():
    p = build_panel(H)
    m = split_masks(p, H)
    X = p[FEATURES].to_numpy(np.float32)
    mu, sd = X[m["train"]].mean(0), X[m["train"]].std(0)
    X = (X - mu) / sd
    y = p[f"y_{H}"].fillna(0).to_numpy(np.float32)
    rows = {k: window_index(p, v, LOOKBACK) for k, v in m.items()}
    return p, X, y, rows


def local_har(p, rows):
    """HAR + VIX regression for each series, fitted on training + validation rows."""
    pred = np.full(len(p), np.nan)
    fit = np.r_[rows["train"], rows["val"]]
    for s, g in p.groupby("series"):
        idx = g.index.to_numpy()
        f = np.intersect1d(idx, fit); t = np.intersect1d(idx, rows["test"])
        if len(f) > 200 and len(t):
            m = LinearRegression().fit(p.loc[f, HAR], p.loc[f, f"y_{H}"])
            pred[t] = m.predict(p.loc[t, HAR])
    return pred


def chronos_forecasts(p, rows, device):
    """Median 5-step forecast of y_5 from the realized history y_5(s), s <= t - 5, of the same series."""
    from chronos import BaseChronosPipeline
    pipe = BaseChronosPipeline.from_pretrained(CHRONOS, device_map=device, torch_dtype=torch.float32)
    y = p[f"y_{H}"].to_numpy(np.float32)
    pos = p.groupby("series").cumcount().to_numpy()
    start = (p.index.to_numpy() - pos)                       # first row of each row's series
    test = rows["test"]
    out = np.full(len(p), np.nan)
    batch = 1024
    t0 = time.time()
    for i in range(0, len(test), batch):
        r = test[i:i + batch]
        ctx = [torch.tensor(y[max(start[j], j - H - 511):j - H + 1]) for j in r]       # realized values only
        q, mean = pipe.predict_quantiles(ctx, prediction_length=H, quantile_levels=[0.5])
        out[r] = q[:, H - 1, 0].float().cpu().numpy()
    return out, time.time() - t0


def chronos2_forecasts(p, rows, device, vix=False):
    """Chronos-2 with the same realized history; with ``vix`` also log VIX as a covariate.

    Log VIX is known up to the forecast day t: its values on the days of the realized
    history are past covariates, those on the days t-4..t, which correspond to the five
    forecast steps, are passed as known future values.
    """
    from chronos import BaseChronosPipeline
    pipe = BaseChronosPipeline.from_pretrained(CHRONOS2, device_map=device, torch_dtype=torch.float32)
    y = p[f"y_{H}"].to_numpy(np.float32)
    v = p["log_vix"].to_numpy(np.float32)
    pos = p.groupby("series").cumcount().to_numpy()
    start = (p.index.to_numpy() - pos)
    test = rows["test"]
    out = np.full(len(p), np.nan)
    batch = 2048
    t0 = time.time()
    for i in range(0, len(test), batch):
        r = test[i:i + batch]
        a = np.maximum(start[r], r - H - 511)
        if vix:
            ctx = [{"target": y[b:j - H + 1], "past_covariates": {"log_vix": v[b:j - H + 1]},
                    "future_covariates": {"log_vix": v[j - H + 1:j + 1]}} for b, j in zip(a, r)]
        else:
            ctx = [torch.tensor(y[b:j - H + 1]) for b, j in zip(a, r)]
        q, _ = pipe.predict_quantiles(ctx, prediction_length=H, quantile_levels=[0.5], batch_size=256)
        out[r] = [float(t[0, H - 1, 0]) for t in q]
    return out, time.time() - t0


CACHE = C.RESULTS_DIR / "lab6g_cache"
TIMED = ["lstm", "gru", "tcn", "transformer", "chronos", "chronos2", "chronos2v"]
_DEV = {"device": "cpu"}


def cached(name, fn, device=None):
    """Results of finished parts are kept on disk, so an interrupted run resumes where it stopped.

    The cache is kept per device: a run on the GPU does not reuse results computed on the CPU
    (they differ in the last digits), and the running times of both devices remain available.
    """
    f = CACHE / f"{name}_{device or _DEV['device']}.npz"
    if f.exists():
        with np.load(f) as z:
            return tuple(z[k] for k in sorted(z.files, key=lambda k: int(k[1:])))
    CACHE.mkdir(parents=True, exist_ok=True)
    out = fn()
    tmp = f.with_name(f.stem + ".tmp.npz")            # written in full and synced before it replaces f
    with open(tmp, "wb") as fh:
        np.savez(fh, **{f"a{i}": np.asarray(v) for i, v in enumerate(out)})
        fh.flush(); os.fsync(fh.fileno())
    tmp.replace(f)
    print(f"{time.strftime('%H:%M:%S')} {name} done", flush=True)
    return out


def main():
    num = {"lookback": LOOKBACK}
    device = large_work_device()
    _DEV["device"] = device
    torch.set_num_threads(16)
    p, X, y, rows = prepare()
    yt = p[f"y_{H}"].to_numpy()
    te = rows["test"]
    num.update({"n_train": len(rows["train"]), "n_val": len(rows["val"]), "n_test": len(te),
                "n_series": p["series"].nunique(), "device": "GPU" if device == "cuda" else "CPU"})
    preds = {"Linear HAR + VIX for each series": local_har(p, rows)}
    pooled = LinearRegression().fit(p.loc[np.r_[rows["train"], rows["val"]], HAR], yt[np.r_[rows["train"], rows["val"]]])
    pp = np.full(len(p), np.nan); pp[te] = pooled.predict(p.loc[te, HAR]); preds["One linear HAR + VIX for all series"] = pp

    # ---------------- an LSTM trained on the S&P 500 only (the local counterpart)
    spx = (p["series"] == "^GSPC").to_numpy()
    def local_lstm():
        loc = []
        for s in SEEDS:
            m, h, win = train_global("lstm", X, y, rows["train"][spx[rows["train"]]], rows["val"][spx[rows["val"]]],
                                     LOOKBACK, seed=s, device=device)
            loc.append(predict_rows(m, win, te[spx[te]], device))
        pl = np.full(len(p), np.nan); pl[te[spx[te]]] = np.mean(loc, 0)
        return (pl,)
    preds["LSTM trained on the S\\&P 500 only"] = cached("local_lstm", local_lstm)[0]

    # ---------------- global models
    info, val_pred = {}, {}
    def global_model(key):
        t0 = time.time(); outs, vouts, hists = [], [], []
        for s in SEEDS:
            m, h, win = train_global(key, X, y, rows["train"], rows["val"], LOOKBACK, seed=s, device=device)
            outs.append(predict_rows(m, win, te, device)); vouts.append(predict_rows(m, win, rows["val"], device))
            hists.append(h)
            print(f"{time.strftime('%H:%M:%S')} {key} seed {s}: {len(h)} epochs", flush=True)
        pg = np.full(len(p), np.nan); pg[te] = np.mean(outs, 0)
        pv = np.full(len(p), np.nan); pv[rows["val"]] = np.mean(vouts, 0)
        return pg, pv, np.array([n_params(m), (time.time() - t0) / 60, np.mean([len(h) for h in hists])])

    for key, name in MODELS.items():
        pg, pv, inf = cached(key, lambda: global_model(key))
        preds[f"Global {name}"] = pg; val_pred[key] = pv
        info[key] = (int(inf[0]), float(inf[1]), float(inf[2]))
        torch.cuda.empty_cache() if device == "cuda" else None

    # ---------------- pretrained forecasters (zero-shot)
    runs = {"chronos": lambda: chronos_forecasts(p, rows, device),
            "chronos2": lambda: chronos2_forecasts(p, rows, device),
            "chronos2v": lambda: chronos2_forecasts(p, rows, device, vix=True)}
    for name, (key, _) in PRETRAINED.items():
        print(f"{time.strftime('%H:%M:%S')} {name}", flush=True)
        try:
            pc, secs = cached(key, runs[key])
            preds[name] = pc
            num[f"{key}_minutes"] = float(secs) / 60
        except Exception as ex:                            # model not downloadable: reported, not hidden
            print(f"{name} unavailable:", ex)

    # ---------------- evaluation
    base = preds["Linear HAR + VIX for each series"]
    ser = p["series"].to_numpy()
    rows_tab = []
    for name, pr in preds.items():
        ok = te[np.isfinite(pr[te]) & np.isfinite(base[te])]
        per = pd.DataFrame({"s": ser[ok], "e": (yt[ok] - pr[ok]) ** 2, "b": (yt[ok] - base[ok]) ** 2}).groupby("s").mean()
        ratio = np.sqrt(per["e"] / per["b"])
        sp = te[spx[te]]
        rows_tab.append((name, rmse(yt[ok], pr[ok]), float(ratio.mean()), int((ratio < 1).sum()), len(ratio),
                         rmse(yt[sp], pr[sp]) if np.isfinite(pr[sp]).all() else np.nan))
    tab = pd.DataFrame(rows_tab, columns=["Model", "Test RMSE", "Mean ratio to HAR", "Series better", "Series",
                                          "S\\&P 500"]).set_index("Model")
    save_table("lab6_global", tab, nd={"Test RMSE": 3, "Mean ratio to HAR": 3, "Series better": 0, "Series": 0, "S\\&P 500": 3},
               col_format="p{5.6cm}ccccc")
    print(tab.to_string())
    sp = te[spx[te]]
    best = min([k for k in preds if k.startswith("Global")], key=lambda k: tab.loc[k, "Mean ratio to HAR"])
    dm, pdm = diebold_mariano(yt[sp] - preds[best][sp], yt[sp] - base[sp], h=H)
    num.update({"best_global": best.replace("Global ", ""), "best_ratio": float(tab.loc[best, "Mean ratio to HAR"]),
                "best_better": int(tab.loc[best, "Series better"]), "best_spx": float(tab.loc[best, "S\\&P 500"]),
                "har_spx": float(tab.iloc[0]["S\\&P 500"]), "pooled_ratio": float(tab.iloc[1]["Mean ratio to HAR"]),
                "local_lstm_spx": float(tab.iloc[2]["S\\&P 500"]), "dm_best_spx": dm, "p_best_spx": pdm,
                **{f"{k}_params": v[0] for k, v in info.items()}, **{f"{k}_minutes": v[1] for k, v in info.items()},
                **{f"{k}_epochs": v[2] for k, v in info.items()},
                **{f"{k}_ratio": float(tab.loc[f'Global {n}', 'Mean ratio to HAR']) for k, n in MODELS.items()}})
    # the same networks trained on the other device, if cached (dropout masks differ between devices)
    other = "cpu" if device == "cuda" else "cuda"
    for key in MODELS:
        fc = CACHE / f"{key}_{other}.npz"
        if fc.exists():
            with np.load(fc) as z:
                po = z["a0"]
            ok = te[np.isfinite(po[te]) & np.isfinite(base[te])]
            per = pd.DataFrame({"s": ser[ok], "e": (yt[ok] - po[ok]) ** 2, "b": (yt[ok] - base[ok]) ** 2}).groupby("s").mean()
            r = np.sqrt(per["e"] / per["b"])
            lab = "gpu" if other == "cuda" else "cpu"
            num[f"{key}_ratio_{lab}"], num[f"{key}_better_{lab}"] = float(r.mean()), int((r < 1).sum())

    # look-ahead check: pooled RMSE ratio to HAR before and after the release of each pretrained model
    d = p["date"].to_numpy()
    for name, (key, release) in PRETRAINED.items():
        if name not in preds:
            continue
        pc = preds[name]
        glob = "global" if key == "chronos" else f"global_{key}"   # the best global network over the same periods
        for lab, msk in (("pre", d[te] < np.datetime64(release)), ("post", d[te] >= np.datetime64(release))):
            r = te[msk & np.isfinite(pc[te]) & np.isfinite(base[te])]
            num[f"{key}_ratio_{lab}"] = rmse(yt[r], pc[r]) / rmse(yt[r], base[r])
            rg = te[msk & np.isfinite(preds[best][te]) & np.isfinite(base[te])]
            num[f"{glob}_ratio_{lab}"] = rmse(yt[rg], preds[best][rg]) / rmse(yt[rg], base[rg])
            num[f"{key}_n_{lab}"] = int(len(r))
        num[f"{key}_ratio"] = float(tab.loc[name, "Mean ratio to HAR"])
        num[f"{key}_better"] = int(tab.loc[name, "Series better"])
        num[f"{key}_spx"] = float(tab.loc[name, "S\\&P 500"])

    # ---------------- figure: ratio to HAR per series and group for the best global model
    ok = te[np.isfinite(preds[best][te]) & np.isfinite(base[te])]
    per = pd.DataFrame({"s": ser[ok], "g": p["group"].to_numpy()[ok], "e": (yt[ok] - preds[best][ok]) ** 2,
                        "b": (yt[ok] - base[ok]) ** 2}).groupby(["g", "s"]).mean()
    per["ratio"] = np.sqrt(per["e"] / per["b"])
    f, axs = P.fig(2.9, ncols=2, gridspec_kw={"width_ratios": [1.2, 1]})
    for k, (g, col) in enumerate(zip(("index", "asset", "industry"), (P.BLUE, P.ORANGE, P.AQUA))):
        v = per.loc[g, "ratio"].to_numpy()
        axs[0].scatter(np.full(len(v), k) + np.random.default_rng(k).uniform(-0.2, 0.2, len(v)), v, s=12, color=col)
    axs[0].axhline(1, color=P.INK2, lw=0.7)
    axs[0].set_xticks(range(3)); axs[0].set_xticklabels(["indices", "other assets", "industries"])
    axs[0].set_ylabel(f"test RMSE of global {best.replace('Global ', '')} / HAR")
    names = [n for n in preds if n != "Linear HAR + VIX for each series"]
    vals = [tab.loc[n, "Mean ratio to HAR"] for n in names]
    axs[1].barh(range(len(names)), vals, color=[P.BLUE if n.startswith("Global") else P.MUTED for n in names])
    axs[1].axvline(1, color=P.INK2, lw=0.7)
    axs[1].set_yticks(range(len(names))); axs[1].set_yticklabels([n.replace("\\&", "&") for n in names], fontsize=7)
    axs[1].invert_yaxis(); axs[1].set_xlabel("mean ratio of test RMSE to HAR"); axs[1].set_xlim(min(vals) - 0.03, max(vals) + 0.03)
    f.tight_layout(); P.save(f, "lab6_global")

    # ---------------- adaptive conformal intervals for the S&P 500 (best global model)
    key = [k for k, n in MODELS.items() if f"Global {n}" == best][0]
    vs = rows["val"][spx[rows["val"]]]
    s_cal = np.abs(yt[vs] - val_pred[key][vs])
    n = len(s_cal)
    q = float(np.quantile(s_cal, np.ceil((n + 1) * 0.9) / n, method="higher"))
    pr, yy = preds[best][sp], yt[sp]
    widths, _ = adaptive_conformal(pr, yy, s_cal, H, 0.1, 0.005, 250)
    cs, ca = np.abs(yy - pr) <= q, np.abs(yy - pr) <= widths
    yrs = p.loc[sp, "date"].dt.year.to_numpy()
    crow = [(str(v), 100 * cs[yrs == v].mean(), 100 * ca[yrs == v].mean()) for v in sorted(set(yrs))]
    crow.append(("2019--2026", 100 * cs.mean(), 100 * ca.mean()))
    save_table("lab6_conformal", pd.DataFrame(crow, columns=["Year", "Coverage, split, \\%", "Coverage, adaptive, \\%"]
                                              ).set_index("Year"), nd=1, col_format="lcc")
    num.update({"cf_split": 100 * cs.mean(), "cf_adapt": 100 * ca.mean(),
                "cf_split_2020": 100 * cs[yrs == 2020].mean(), "cf_adapt_2020": 100 * ca[yrs == 2020].mean(),
                "cf_halfwidth": q})
    # running times on both devices, where a run on that device is cached (seconds and minutes as stored)
    for key in TIMED:
        for dev in ("cpu", "cuda"):
            fc = CACHE / f"{key}_{dev}.npz"
            if fc.exists():
                with np.load(fc) as z:
                    a = [z[k] for k in sorted(z.files, key=lambda k: int(k[1:]))]
                mins = float(a[1]) / 60 if key.startswith("chronos") else float(a[-1][1])
                num[f"{key}_minutes_{'gpu' if dev == 'cuda' else 'cpu'}"] = mins
    save_numbers("lab6g", num)
    print(num)


if __name__ == "__main__":
    main()
