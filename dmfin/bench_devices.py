"""When does a graphics processor pay off?  Time of the PyTorch workloads of the labs on CPU and GPU.

Every workload is timed with the sizes, batches and CPU threads that its lab uses, after a
warm-up step (GPU kernels are compiled or loaded on first use).  The result is the table of
devices in the introduction to the practical part and the rule in ``dmfin/device.py``.
Without a GPU only the CPU column is filled.

Run:  python -m dmfin.bench_devices
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import torch
from torch import nn

from dmfin.device import gpu_available
from dmfin.reporting import save_numbers, save_table


def _sync(dev):
    if dev == "cuda":
        torch.cuda.synchronize()


def _timed(fn, dev, reps=1):
    fn(warm=True)                                         # first use: kernels, caches
    _sync(dev)
    t = time.perf_counter()
    for _ in range(reps):
        fn(warm=False)
    _sync(dev)
    return (time.perf_counter() - t) / reps


def mlp_epoch(dev, n=18_000, d=23, width=64, depth=2, batch=256):
    """Lab 3: perceptron on the credit clients (shape of the data, random values)."""
    from lab3_classification.mlp import MLP
    g = torch.Generator().manual_seed(0)
    X = torch.randn(n, d, generator=g).to(dev); y = (torch.rand(n, generator=g) < 0.22).float().to(dev)
    m = MLP(d, width, depth).to(dev); opt = torch.optim.AdamW(m.parameters(), 1e-3); lf = nn.BCEWithLogitsLoss()

    def fn(warm):
        for i in range(0, batch * 2 if warm else n, batch):
            opt.zero_grad(); lf(m(X[i:i + batch]), y[i:i + batch]).backward(); opt.step()
    return _timed(fn, dev)


def rnn_epoch(dev, hidden, layers, batch, cell="lstm", autoencoder=False):
    """Labs 6 and 8: one epoch of the forecaster (or autoencoder) on the S&P 500 windows."""
    from lab6_rnn.common import lab6_frame
    from lab6_rnn.nets import INPUTS, LSTMAutoencoder, SeqData, SeqNet, train
    df, _ = lab6_frame()
    sd = SeqData(df, "y_logrv_5", lookback=60, horizon=5)
    X, y, Xv, yv = sd["train"].X, sd["train"].y, sd["val"].X, sd["val"].y
    make = (lambda: LSTMAutoencoder(len(INPUTS), hidden=hidden, latent=8)) if autoencoder else \
        (lambda: SeqNet(len(INPUTS), hidden, layers, cell, dropout=0.1 if layers > 1 else 0.0))
    loss = "recon" if autoencoder else "mse"

    def fn(warm):
        n = batch * 4 if warm else len(X)
        train(make(), X[:n], None if autoencoder else y[:n], Xv[:batch], None if autoencoder else yv[:batch],
              loss=loss, batch=batch, max_epochs=1, patience=1, device=dev)
    return _timed(fn, dev)


def dqn_updates(dev, n=1000, d=24, actions=2, hidden=64, batch=64):
    """Lab 7: 1 000 updates of the Q-network on minibatches of the replay buffer."""
    net = nn.Sequential(nn.Linear(d, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(),
                        nn.Linear(hidden, actions)).to(dev)
    opt = torch.optim.Adam(net.parameters(), 1e-3)
    S = torch.randn(5000, d).to(dev); R = torch.randn(5000).to(dev); A = torch.randint(0, actions, (5000,)).to(dev)
    g = torch.Generator().manual_seed(0)

    def fn(warm):
        for _ in range(10 if warm else n):
            b = torch.randint(0, 5000, (batch,), generator=g).to(dev)
            q = net(S[b]).gather(1, A[b, None]).squeeze(1)
            loss = nn.functional.mse_loss(q, R[b] + 0.9 * net(S[b]).max(1).values.detach())
            opt.zero_grad(); loss.backward(); opt.step()
    return _timed(fn, dev)


def search_epoch(dev):
    """Lab 8 searches: one epoch of the global perceptron of the default configuration."""
    from lab8_genetic import search as S
    d = _data("search", S.Data)
    cfg = S.DEFAULT
    X, y = d.X.to(dev), d.y.to(dev)
    offs = torch.arange(-cfg["lookback"] + 1, 1, device=dev)
    torch.manual_seed(0)
    net = S.MLP(9 * cfg["lookback"], cfg["width"], cfg["layers"], cfg["dropout"]).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    r = torch.as_tensor(d.rows["train"])
    perm = r[torch.randperm(len(r), generator=torch.Generator().manual_seed(0))].to(dev)

    def fn(warm):
        for i in range(0, cfg["batch"] * 2 if warm else len(perm), cfg["batch"]):
            b = perm[i:i + cfg["batch"]]
            loss = nn.functional.mse_loss(net(X[b[:, None] + offs].reshape(len(b), -1)), y[b])
            opt.zero_grad(); loss.backward(); opt.step()
    return _timed(fn, dev)


_CACHE: dict = {}


def _data(key, make):
    if key not in _CACHE:
        _CACHE[key] = make()
    return _CACHE[key]


def global_epoch(dev, name):
    """Lab 6: one epoch of a global network on the 62-series panel (batches of 1 024 windows)."""
    from lab6_rnn import run_global as G
    from lab6_rnn.global_models import train_global
    p, X, y, rows = _data("panel", G.prepare)

    def fn(warm):
        tr = rows["train"][:4096] if warm else rows["train"]
        train_global(name, X, y, tr, rows["val"][:4096], G.LOOKBACK, seed=0, device=dev, max_epochs=1)
    return _timed(fn, dev)


def chronos2_batch(dev, n=2048):
    """Lab 6: zero-shot forecasts of Chronos-2 with VIX for 2 048 test days."""
    from lab6_rnn import run_global as G
    p, X, y, rows = _data("panel", G.prepare)
    sub = {"test": rows["test"][:n]}

    def fn(warm):
        G.chronos2_forecasts(p, {"test": sub["test"][:64]} if warm else sub, dev, vix=True)
    return _timed(fn, dev)


def finetune_steps(dev, steps=20, batch=32):
    """Lab 3: fine-tuning of the pretrained DistilRoBERTa on tweets (20 updates of 32 tweets)."""
    from lab3_classification import run_text as R
    from lab3_classification import text as T
    tr, va, te, _ = _data("tweets", R.data)
    ids = _data("tweet_ids", lambda: T.encode(T.tokenizer(), tr["text"][:steps * batch]))
    y = tr["label"].to_numpy()[:steps * batch]
    m = T.transformer(True, 0)

    def fn(warm):
        n = batch * (2 if warm else steps)
        T.train(m, ids[:n], y[:n], ids[:batch], y[:batch], lr=3e-5, max_epochs=1, batch=batch,
                device=dev)
    return _timed(fn, dev)


WORKLOADS = [  # name, lab, unit, CPU threads used by the lab, function of the device
    ("Perceptron, two layers of 64 units, batch 256", "3", "epoch", 1, mlp_epoch),
    ("Q-network, two layers of 64 units, batch 64", "7", "1\\,000 updates", 1, dqn_updates),
    ("LSTM, 32 units, batch 64", "6", "epoch", 2, lambda dev: rnn_epoch(dev, 32, 1, 64)),
    ("GRU, 32 units, batch 64", "6", "epoch", 2, lambda dev: rnn_epoch(dev, 32, 1, 64, cell="gru")),
    ("LSTM autoencoder, 32 units, batch 64", "6", "epoch", 16, lambda dev: rnn_epoch(dev, 32, 1, 64, autoencoder=True)),
    ("LSTM, two layers of 128 units, batch 32", "8", "epoch", 2, lambda dev: rnn_epoch(dev, 128, 2, 32)),
    ("LSTM, two layers of 128 units, batch 128", "8", "epoch", 2, lambda dev: rnn_epoch(dev, 128, 2, 128)),
    ("Global perceptron of the searches, batch 1\\,024", "8", "epoch", 1, search_epoch),
    ("Global LSTM, 62 series, batch 1\\,024", "6", "epoch", 16, lambda dev: global_epoch(dev, "lstm")),
    ("Global Transformer, 62 series, batch 1\\,024", "6", "epoch", 16, lambda dev: global_epoch(dev, "transformer")),
    ("Chronos-2 with VIX, zero-shot", "6", "2\\,048 forecasts", 16, chronos2_batch),
    ("DistilRoBERTa fine-tuning, batch 32", "3", "20 updates", 16, finetune_steps),
]


def main():
    gpu = gpu_available()
    rows, num = [], {"gpu": "yes" if gpu else "no"}
    for i, (name, lab, unit, threads, fn) in enumerate(WORKLOADS):
        torch.set_num_threads(threads)
        cpu = fn("cpu")
        gp = fn("cuda") if gpu else np.nan
        rows.append((name, lab, unit, threads, cpu, gp, cpu / gp if gpu else np.nan))
        print(f"{name:50s} CPU({threads}) {cpu:8.3f} s  GPU {gp:8.3f} s", flush=True)
    tab = pd.DataFrame(rows, columns=["Workload", "Lab", "Unit", "CPU threads", "CPU, s", "GPU, s", "CPU / GPU"])
    num.update({f"w{i}_{k}": float(v) for i, r in enumerate(rows) for k, v in (("cpu", r[4]), ("gpu", r[5]), ("ratio", r[6]))
                if np.isfinite(v)})
    save_table("devices_bench", tab.set_index("Workload"),
               nd={"CPU threads": 0, "CPU, s": 2, "GPU, s": 2, "CPU / GPU": 1}, col_format="p{5.2cm}clcccc")
    save_numbers("devb", num)
    print(tab.to_string())


if __name__ == "__main__":
    main()
