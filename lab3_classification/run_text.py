"""Lab 3 (iteration 3). Sentiment of financial news tweets: where a pretrained transformer gains strongly.

Data (D8): the published training tweets (9 543) and validation tweets (2 388) of financial news accounts, labelled
bearish, bullish or neutral.  The published validation tweets are the test set, without those that repeat a training
tweet once links are removed; 15 % of the training tweets (stratified) are the validation set for early stopping and
for the hyperparameters, which every model chooses from a small grid by validation macro-F1.

A. All training tweets: majority class; naive Bayes and TF-IDF + logistic regression (CPU); an LSTM and DistilRoBERTa
   trained from random weights; the pretrained DistilRoBERTa as fixed features of a logistic regression and fine-tuned
   (GPU if there is one; three seeds each).  Paired bootstrap of the gain in macro-F1.
B. Learning curves: 250 to 4 000 training tweets, three subsamples each.
C. Errors: confusion matrices and examples.

CPU and GPU work run one after the other (dmfin.device).  The results of the networks are cached in
results/lab3_text_cache/, so an interrupted run resumes where it stopped.

Run:  python -m lab3_classification.run_text
"""
from __future__ import annotations

import os
import re
import time

import matplotlib.ticker
import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from dmfin import config as C
from dmfin import plotting as P
from dmfin.data_ext import TWEET_LABELS, load_fin_tweets
from dmfin.device import large_work_device, rnn_device
from dmfin.reporting import save_numbers, save_table

from . import text as T

SEEDS = [0, 1, 2]
SIZES = [250, 500, 1000, 2000, 4000]
CACHE = C.RESULTS_DIR / "lab3_text_cache"
GRIDS = {   # learning rates, maximum epochs, patience, batch
    "lstm": dict(lrs=[3e-4, 1e-3, 3e-3], max_epochs=30, patience=4, batch=32),
    "scratch": dict(lrs=[3e-5, 1e-4, 3e-4], max_epochs=20, patience=3, batch=32),
    "finetune": dict(lrs=[1e-5, 3e-5, 1e-4], max_epochs=6, patience=2, batch=32),
}
NAMES = {"majority": "Majority class (neutral)", "nb": "Naive Bayes, word counts",
         "tfidf": "Logistic regression, TF-IDF of words and characters",
         "lstm": "LSTM, random initial weights", "scratch": "DistilRoBERTa, random initial weights",
         "frozen": "Logistic regression on pretrained DistilRoBERTa features",
         "finetune": "DistilRoBERTa, pretrained and fine-tuned"}


def data():
    tr, te = load_fin_tweets()
    tr["text"], te["text"] = tr["text"].map(T.normalize), te["text"].map(T.normalize)
    key = lambda s: s.str.lower()
    dup = key(te["text"]).isin(set(key(tr["text"])))
    te = te[~dup].reset_index(drop=True)
    tr, va = train_test_split(tr, test_size=0.15, stratify=tr["label"], random_state=C.SEED)
    return tr.reset_index(drop=True), va.reset_index(drop=True), te, int(dup.sum())


def cached(name, fn):
    """Arrays returned by ``fn`` are stored in results/lab3_text_cache/<name>.npz (written in full, then renamed)."""
    f = CACHE / f"{name}.npz"
    if f.exists():
        with np.load(f) as z:
            return {k: z[k] for k in z.files}
    CACHE.mkdir(parents=True, exist_ok=True)
    out = fn()
    tmp = f.with_name(f.stem + ".tmp.npz")
    with open(tmp, "wb") as fh:
        np.savez(fh, **out)
        fh.flush(); os.fsync(fh.fileno())
    tmp.replace(f)
    print(f"{time.strftime('%H:%M:%S')} {name} done", flush=True)
    return out


def scores(y, p):
    pred = p.argmax(1) if p.ndim == 2 else p
    f1 = [2 * np.sum((pred == k) & (y == k)) / max(1, np.sum(pred == k) + np.sum(y == k)) for k in range(3)]
    return {"acc": float(np.mean(pred == y)), "f1": T.macro_f1(y, pred), "f1_bear": f1[0], "f1_bull": f1[1],
            "f1_neut": f1[2]}


# ----------------------------------------------------------------------------- CPU: bag of words
def bag_of_words(tr, va, te, sub=None, models=("nb", "tfidf")):
    """Hyperparameters by validation macro-F1; returns per model: test probabilities, parameters, seconds, choice.
    The vocabulary and the weights of the terms are learned from the training tweets only."""
    X, y = (tr["text"], tr["label"].to_numpy()) if sub is None else (tr["text"].iloc[sub], tr["label"].to_numpy()[sub])
    yv = va["label"].to_numpy()
    out = {}
    for key in models:
        t0 = time.perf_counter()
        vec = (CountVectorizer(ngram_range=(1, 2)) if key == "nb" else T.tfidf_features()).fit(X)
        A, V, E = vec.transform(X), vec.transform(va["text"]), vec.transform(te["text"])
        t_vec = time.perf_counter() - t0
        if key == "nb":
            grid, make = [(a,) for a in (0.03, 0.1, 0.3, 1.0)], lambda a: MultinomialNB(alpha=a)
        else:
            grid = [(c, w) for c in (1, 3, 10, 30, 100) for w in (None, "balanced")]
            make = lambda c, w: LogisticRegression(C=c, class_weight=w, max_iter=5000)
        best = max(grid, key=lambda g: T.macro_f1(yv, make(*g).fit(A, y).predict(V)))
        t0 = time.perf_counter()
        m = make(*best).fit(A, y)
        npar = m.coef_.size + m.intercept_.size if key == "tfidf" else m.feature_log_prob_.size + 3
        out[key] = dict(test=m.predict_proba(E), val=m.predict_proba(V), params=npar,
                        sec=t_vec + time.perf_counter() - t0, choice=best)
    return out


def frozen_logistic(E, y, yv, sub=None):
    """Logistic regression on the fixed sentence vectors of the pretrained encoder; C and weights by validation."""
    Etr, ytr = (E["train"], y) if sub is None else (E["train"][sub], y[sub])
    sc = StandardScaler().fit(Etr)
    Z = {k: sc.transform(E[k]) for k in ("train", "val", "test")}
    Ztr = Z["train"] if sub is None else Z["train"][sub]
    grid = [(c, w) for c in (0.003, 0.01, 0.03, 0.1, 1.0) for w in (None, "balanced")]
    fit = lambda c, w: LogisticRegression(C=c, class_weight=w, max_iter=5000).fit(Ztr, ytr)
    best = max(grid, key=lambda g: T.macro_f1(yv, fit(*g).predict(Z["val"])))
    t0 = time.perf_counter()
    m = fit(*best)
    return dict(test=m.predict_proba(Z["test"]), val=m.predict_proba(Z["val"]), params=m.coef_.size + 3,
                sec=time.perf_counter() - t0, choice=best)


# ----------------------------------------------------------------------------- GPU: networks
def network(kind, ids, y, yv, seed, lr, sub=None, max_epochs=None, patience=None):
    """One network: test and validation probabilities, history, seconds, number of parameters."""
    g = GRIDS[kind]
    Xtr, ytr = ids["train"], y
    if sub is not None:
        Xtr, ytr = [Xtr[i] for i in sub], y[sub]
    if kind == "lstm":
        model, dev, pad_id = T.LSTMClassifier(ids["vocab"].size), rnn_device(128), 0
        Xtr, Xva, Xte = ids["vocab"](Xtr), ids["vocab"](ids["val"]), ids["vocab"](ids["test"])
    else:
        model, dev, pad_id = T.transformer(kind == "finetune", seed), large_work_device(), ids["pad"]
        Xva, Xte = ids["val"], ids["test"]
    model, hist, sec = T.train(model, Xtr, ytr, Xva, yv, lr=lr, max_epochs=max_epochs or g["max_epochs"],
                               patience=patience or g["patience"], batch=g["batch"], seed=seed, device=dev, pad_id=pad_id)
    out = dict(test=T.predict_proba(model, Xte, dev, pad_id), val=T.predict_proba(model, Xva, dev, pad_id), hist=hist,
               sec=np.array(sec), params=np.array(T.n_params(model)))
    del model
    if dev == "cuda":
        torch.cuda.empty_cache()
    return out


def lc_epochs(n, batch=32):
    """Small training sets get more epochs, so that every run makes about 1 500 updates (at most 30 epochs)."""
    return int(min(30, max(GRIDS["finetune"]["max_epochs"], np.ceil(1500 / np.ceil(n / batch)))))


def main():
    num = {}
    tr, va, te, n_dup = data()
    y, yv, yt = (d["label"].to_numpy() for d in (tr, va, te))
    full = pd.concat([tr, va])
    num.update({"n_published": len(full) + len(te) + n_dup, "n_train": len(tr), "n_val": len(va), "n_test": len(te),
                "n_dup": n_dup, **{f"share_{l}": 100 * np.mean(np.r_[y, yv, yt] == k) for k, l in enumerate(TWEET_LABELS)},
                "words_median": int(full["text"].str.split().str.len().median()),
                "words_max": int(full["text"].str.split().str.len().max())})
    rng = np.random.default_rng(C.SEED)
    subs = {(n, s): np.sort(rng.choice(len(tr), n, replace=False)) for n in SIZES for s in SEEDS}

    # ---- phase 1, CPU: bag of words (all tweets and the learning-curve subsamples)
    torch.set_num_threads(4 if large_work_device() == "cuda" else os.cpu_count())
    res = {}
    cpu = threadpool_limits(4)                  # small problems: a few threads are faster than all cores
    bow = bag_of_words(tr, va, te)
    res.update(bow)
    lc = {("tfidf", n, s): bag_of_words(tr, va, te, subs[(n, s)], ("tfidf",))["tfidf"]["test"] for n, s in subs}
    num.update({"nb_alpha": f'{bow["nb"]["choice"][0]:g}', "tfidf_c": bow["tfidf"]["choice"][0],
                "tfidf_weights": "balanced" if bow["tfidf"]["choice"][1] else "equal",
                "tfidf_features": int(res["tfidf"]["params"] // 3)})

    # ---- phase 2, GPU (if there is one): token ids, networks, sentence vectors of the pretrained encoder
    tok = T.tokenizer()
    ids = {k: T.encode(tok, d["text"]) for k, d in (("train", tr), ("val", va), ("test", te))}
    ids["pad"] = tok.pad_token_id
    ids["vocab"] = T.Vocabulary(ids["train"], len(tok))
    num.update({"tokens_median": int(np.median([len(s) for s in ids["train"]])), "vocab_lstm": ids["vocab"].size,
                "truncated": int(sum(len(x) > T.MAX_LEN for x in T.encode(tok, full["text"], 512)))})
    runs, chosen = {}, {}
    for kind, g in GRIDS.items():
        grid = {lr: cached(f"{kind}_lr{lr:g}_s0", lambda: network(kind, ids, y, yv, 0, lr)) for lr in g["lrs"]}
        lr = max(grid, key=lambda r: grid[r]["hist"][:, 3].max())
        chosen[kind] = lr
        runs[kind] = [grid[lr]] + [cached(f"{kind}_lr{lr:g}_s{s}", lambda: network(kind, ids, y, yv, s, lr))
                                   for s in SEEDS[1:]]
        num.update({f"{kind}_lr": f"{lr:g}".replace("e-0", "\\cdot 10^{-") + ("}" if "e-0" in f"{lr:g}" else ""),
                    f"{kind}_epochs": f'{np.mean([r["hist"][:, 3].argmax() + 1 for r in runs[kind]]):.1f}',
                    f"{kind}_sec": f'{np.mean([r["sec"] for r in runs[kind]]):.0f}'})
    for n, s in subs:
        lc[("finetune", n, s)] = cached(f"lc_finetune_n{n}_s{s}", lambda: network(
            "finetune", ids, y, yv, s, chosen["finetune"], sub=subs[(n, s)], max_epochs=lc_epochs(n), patience=4))["test"]

    def sentence_vectors():
        from transformers import AutoModel
        enc = AutoModel.from_pretrained(T.LM, revision=T.LM_REV)
        t0 = time.perf_counter()
        out = {k: T.embed(enc, ids[k], large_work_device(), ids["pad"]) for k in ("train", "val", "test")}
        return {**out, "sec": np.array(time.perf_counter() - t0)}
    E = cached("sentence_vectors", sentence_vectors)
    device = large_work_device()
    num["device"] = "GPU" if device == "cuda" else "CPU"

    # ---- phase 3, CPU: logistic regression on the fixed vectors, evaluation
    res["frozen"] = frozen_logistic(E, y, yv)
    res["frozen"]["sec"] += float(E["sec"])
    for n, s in subs:
        lc[("frozen", n, s)] = frozen_logistic(E, y, yv, subs[(n, s)])["test"]
    res["majority"] = dict(test=np.tile(np.eye(3)[np.bincount(y).argmax()], (len(yt), 1)), params=1, sec=0.0)

    rows, f1_seeds = [], {}
    for key, name in NAMES.items():
        if key in runs:
            sc = [scores(yt, r["test"]) for r in runs[key]]
            m = {k: np.mean([d[k] for d in sc]) for k in sc[0]}
            sd, npar, sec = np.std([d["f1"] for d in sc]), int(runs[key][0]["params"]), np.mean([r["sec"] for r in runs[key]])
            f1_seeds[key] = [d["f1"] for d in sc]
        else:
            m, sd, npar, sec = scores(yt, res[key]["test"]), np.nan, int(res[key]["params"]), float(res[key]["sec"])
        rows.append((name, npar, m["acc"], m["f1"], sd, m["f1_bear"], m["f1_bull"], sec))
        num.update({f"{key}_acc": m["acc"], f"{key}_f1": m["f1"], f"{key}_f1_bear": m["f1_bear"],
                    f"{key}_f1_bull": m["f1_bull"], f"{key}_params": npar})
        if np.isfinite(sd):
            num[f"{key}_f1_sd"] = sd
    tab = pd.DataFrame(rows, columns=["Model", "Parameters fitted", "Accuracy", "Macro-F1", "SD", "F1 bearish",
                                      "F1 bullish", "Time, s"]).set_index("Model")
    save_table("lab3_text_results", tab, nd={"Parameters fitted": 0, "Accuracy": 3, "Macro-F1": 3, "SD": 3,
                                             "F1 bearish": 3, "F1 bullish": 3, "Time, s": 1},
               col_format="p{4.0cm}ccccccc")
    pred_tfidf = res["tfidf"]["test"].argmax(1)
    pred_ft = np.array([r["test"].argmax(1) for r in runs["finetune"]])
    (lo, hi), p_boot = T.paired_bootstrap(yt, pred_tfidf, pred_ft, n=2000, seed=C.SEED)
    err_tfidf, err_ft = 1 - num["tfidf_acc"], 1 - num["finetune_acc"]
    num.update({"gain_f1": num["finetune_f1"] - num["tfidf_f1"], "gain_lo": lo, "gain_hi": hi, "gain_p": p_boot,
                "err_cut": f"{100 * (1 - err_ft / err_tfidf):.0f}", "params_m": int(runs["finetune"][0]["params"]) / 1e6,
                "frozen_sec": f'{float(E["sec"]):.1f}', "finetune_min_f1": float(np.min(f1_seeds["finetune"]))})

    # ---- B. learning curves
    curves = {}
    for key, label in (("tfidf", "TF-IDF + logistic regression"), ("frozen", "pretrained features + logistic regression"),
                       ("finetune", "pretrained, fine-tuned")):
        pts = []
        for n in SIZES:
            f = [T.macro_f1(yt, lc[(key, n, s)].argmax(1)) for s in SEEDS]
            pts.append((n, np.mean(f), np.std(f)))
        allf = f1_seeds.get(key, [num[f"{key}_f1"]])
        pts.append((len(tr), np.mean(allf), np.std(allf)))
        curves[label] = np.array(pts)
        for n, mu, _ in pts[:-1]:
            num[f"lc_{key}_{n}"] = mu
    f, ax = P.fig(2.7)
    for (label, v), col in zip(curves.items(), (P.ORANGE, P.AQUA, P.BLUE)):
        ax.errorbar(v[:, 0], v[:, 1], yerr=v[:, 2], fmt="o-", ms=3, lw=1.1, color=col, label=label, capsize=2)
    ax.set_xscale("log"); ax.set_xlabel("training tweets"); ax.set_ylabel("test macro-F1")
    ax.set_xticks(SIZES + [len(tr)], [f"{n:,}".replace(",", "\u2009") for n in SIZES + [len(tr)]])
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    h, l = ax.get_legend_handles_labels()
    ax.legend(h[::-1], l[::-1], fontsize=9)                    # in the order of the curves, top first
    P.save(f, "lab3_text_learning_curves")
    num["lc_n_equal"] = int(next((n for n in SIZES if num[f"lc_finetune_{n}"] >= num["tfidf_f1"]), len(tr)))

    # ---- C. errors: confusion matrices (rows: true class, %), examples
    f, axs = P.fig(2.6, ncols=2)
    for ax, (title, preds) in zip(axs, (("TF-IDF + logistic regression", pred_tfidf[None]),
                                        ("DistilRoBERTa, fine-tuned (3 seeds)", pred_ft))):
        cm = np.mean([np.bincount(3 * yt + p, minlength=9).reshape(3, 3) for p in preds], axis=0)
        cm = 100 * cm / cm.sum(1, keepdims=True)
        ax.imshow(cm, cmap=P.SEQ_CELL, vmin=0, vmax=100); ax.grid(False)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{cm[i, j]:.0f}", ha="center", va="center", fontsize=9, color=P.INK)
        ax.set_xticks(range(3)); ax.set_xticklabels(TWEET_LABELS); ax.set_yticks(range(3)); ax.set_yticklabels(TWEET_LABELS)
        ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title(title, fontsize=9)
    f.tight_layout(); P.save(f, "lab3_text_confusion")
    bull_as_bear = lambda p: np.mean(p[yt == 1] == 0) + np.mean(p[yt == 0] == 1)
    num.update({"opposite_tfidf": f"{100 * bull_as_bear(pred_tfidf) / 2:.1f}",
                "opposite_ft": f"{100 * np.mean([bull_as_bear(p) for p in pred_ft]) / 2:.1f}"})

    ok = te["text"].map(lambda s: s.isascii() and "URL" not in s and len(s.split()) <= 14).to_numpy()
    right_ft, wrong_ft = (pred_ft == yt).all(0), (pred_ft != yt).all(0)
    pick = np.random.default_rng(C.SEED)
    a = pick.choice(np.flatnonzero(ok & right_ft & (pred_tfidf != yt)), 5, replace=False)
    b = pick.choice(np.flatnonzero(ok & wrong_ft), 3, replace=False)
    lab = lambda k: TWEET_LABELS[k]
    ex = pd.DataFrame([(latex_text(te["text"][i]), lab(yt[i]), lab(pred_tfidf[i]),
                        lab(np.bincount(pred_ft[:, i], minlength=3).argmax())) for i in np.r_[a, b]],
                      columns=["Tweet (test set)", "Label", "TF-IDF", "Fine-tuned"])
    save_table("lab3_text_examples", ex, index=False, col_format="p{9.4cm}ccc")
    num.update({"n_right_ft_wrong_tfidf": int(np.sum(right_ft & (pred_tfidf != yt))),
                "n_wrong_ft_right_tfidf": int(np.sum(wrong_ft & (pred_tfidf == yt))), "n_wrong_ft": int(wrong_ft.sum())})
    save_numbers("lab3t", num)
    print(tab.to_string()); print(num)


def latex_text(s: str) -> str:
    rep = {"\\": r"\textbackslash{}", "{": r"\{", "}": r"\}", "$": r"\$", "&": r"\&", "%": r"\%", "#": r"\#",
           "_": r"\_", "^": r"\textasciicircum{}", "~": r"\textasciitilde{}"}
    return re.sub(r"[\\{}$&%#_^~]", lambda m: rep[m.group(0)], s)


if __name__ == "__main__":
    main()
