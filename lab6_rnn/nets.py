"""Lab 6. Recurrent networks for financial time series (PyTorch).

* ``build_inputs``  - daily input channels and targets;
* ``SeqData``       - leakage-free windows for the train / validation / test blocks;
* ``SeqNet``        - Elman RNN, LSTM or GRU encoder + linear head
                      (regression or binary classification);
* ``LSTMAutoencoder`` - sequence-to-sequence reconstruction for anomaly
                      detection and motif embeddings;
* ``train``         - mini-batch Adam with early stopping on validation loss,
                      on the CPU or on a GPU (CUDA or ROCm).
"""
from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch import nn

from dmfin.splits import chrono_split, make_windows

INPUTS = ["ret", "log_pk_d", "log_vix", "d_log_vix", "term_10y3m", "credit_baa_aaa",
          "volume_z", "d_dgs10"]


def build_inputs(feats: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    d = pd.DataFrame(index=feats.index)
    d["ret"] = feats["ret"]
    d["log_pk_d"] = np.log(feats["pk_d"].clip(lower=1e-3))
    d["log_vix"] = np.log(feats["vix"])
    d["d_log_vix"] = d["log_vix"].diff()
    d["term_10y3m"] = feats["term_10y3m"]
    d["credit_baa_aaa"] = feats["credit_baa_aaa"]
    d["volume_z"] = feats["volume_z"]
    d["d_dgs10"] = panel["DGS10"].diff()
    # HAR regressors (for the baseline) and targets
    for c in ("pk_d", "pk_w", "pk_m"):
        d[f"har_{c}"] = np.log(feats[c].clip(lower=1e-3))
    d["log_rv_5_past"] = np.log(np.sqrt(252 / 5 * (feats["ret"] ** 2).rolling(5).sum()).clip(lower=1e-3))
    for col in ("y_logrv_5", "y_rv_5", "y_ret_5", "y_logrv_1"):
        d[col] = feats[col]
    return d.dropna()


@dataclass
class Block:
    X: np.ndarray
    y: np.ndarray
    dates: pd.DatetimeIndex
    pos: np.ndarray


class SeqData:
    """Windows of ``lookback`` days ending at t; label = target[t].

    The scaler is fitted on training rows only.  Labels whose horizon would
    cross the end of their block are purged.  Inputs may reach back into
    the previous block (that history is observable at time t).
    """

    def __init__(self, df: pd.DataFrame, target: str, lookback: int = 60, horizon: int = 5,
                 inputs=INPUTS, y_transform=None):
        self.df, self.inputs, self.lookback, self.horizon = df, list(inputs), lookback, horizon
        sp = chrono_split(df.index, horizon=horizon)
        idx = df.index
        tr_rows = idx.isin(sp.train)
        mu = df.loc[tr_rows, self.inputs].mean(); sd = df.loc[tr_rows, self.inputs].std()
        self.mu, self.sd = mu, sd
        V = ((df[self.inputs] - mu) / sd).to_numpy(np.float32)
        y = df[target].to_numpy(np.float32)
        if y_transform is not None:
            y = y_transform(y)
        self.V, self.yall = V, y
        self.blocks = {}
        for name, dates in (("train", sp.train), ("val", sp.val), ("test", sp.test)):
            pos = np.where(idx.isin(dates))[0]
            X, yy, pos = make_windows(V, y, pos, lookback)
            self.blocks[name] = Block(X, yy, idx[pos], pos)

    def __getitem__(self, name) -> Block:
        return self.blocks[name]


class SeqNet(nn.Module):
    def __init__(self, n_in: int, hidden: int = 32, layers: int = 1, cell: str = "lstm",
                 dropout: float = 0.0, out: int = 1):
        super().__init__()
        rnn = {"rnn": nn.RNN, "lstm": nn.LSTM, "gru": nn.GRU}[cell]
        self.rnn = rnn(n_in, hidden, num_layers=layers, batch_first=True,
                       dropout=dropout if layers > 1 else 0.0)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, out)

    def forward(self, x):
        h, _ = self.rnn(x)
        return self.head(self.drop(h[:, -1])).squeeze(-1)


class LSTMAutoencoder(nn.Module):
    """Encoder LSTM -> latent vector -> repeated -> decoder LSTM -> linear."""

    def __init__(self, n_in: int, hidden: int = 32, latent: int = 8):
        super().__init__()
        self.enc = nn.LSTM(n_in, hidden, batch_first=True)
        self.to_z = nn.Linear(hidden, latent)
        self.from_z = nn.Linear(latent, hidden)
        self.dec = nn.LSTM(hidden, hidden, batch_first=True)
        self.out = nn.Linear(hidden, n_in)

    def encode(self, x):
        h, _ = self.enc(x)
        return self.to_z(h[:, -1])

    def forward(self, x):
        z = self.encode(x)
        hz = self.from_z(z).unsqueeze(1).repeat(1, x.shape[1], 1)
        h, _ = self.dec(hz)
        return self.out(h)


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)


from dmfin.device import gpu_available  # noqa: E402,F401  (one rule for all labs, see dmfin/device.py)


def gpu_rnn_check() -> str | None:
    """One training step of a two-layer GRU with dropout on the GPU; None if it works.

    On ROCm, MIOpen compiles its RNN-dropout kernel at run time and needs the
    rocRAND headers (Debian/Ubuntu: ``sudo apt install librocrand-dev``).
    """
    try:
        m = nn.GRU(4, 8, 2, dropout=0.1, batch_first=True).cuda()
        m(torch.randn(3, 5, 4, device="cuda"))[0].sum().backward()
        return None
    except RuntimeError as e:
        return str(e).strip().splitlines()[0]


def _device_of(model: nn.Module) -> torch.device:
    return next(model.parameters()).device


def train(model: nn.Module, Xtr, ytr, Xva, yva, loss: str = "mse", lr: float = 1e-3,
          batch: int = 64, max_epochs: int = 100, patience: int = 10, weight_decay: float = 0.0,
          seed: int = 0, verbose: bool = False, clip: float = 1.0, device: str = "cpu"):
    """Adam + early stopping; returns (best model, history of (train, val) losses).

    The model and the data are moved to ``device`` once.  The mini-batches are
    drawn by a CPU generator, so both devices see the same sequence of batches;
    the results still differ in the last digits because the kernels differ.
    """
    set_seed(seed)
    dev = torch.device(device)
    model.to(dev)
    lossf = {"mse": nn.MSELoss(), "bce": nn.BCEWithLogitsLoss(), "recon": nn.MSELoss()}[loss]
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    Xtr_t, Xva_t = torch.as_tensor(Xtr, device=dev), torch.as_tensor(Xva, device=dev)
    ytr_t = Xtr_t if loss == "recon" else torch.as_tensor(ytr, device=dev)
    yva_t = Xva_t if loss == "recon" else torch.as_tensor(yva, device=dev)
    best, best_state, wait, hist = np.inf, None, 0, []
    g = torch.Generator().manual_seed(seed)
    for ep in range(max_epochs):
        model.train()
        perm = torch.randperm(len(Xtr_t), generator=g).to(dev)
        tot = torch.zeros((), dtype=torch.float64, device=dev)   # no host sync inside the epoch
        for i in range(0, len(perm), batch):
            b = perm[i:i + batch]
            opt.zero_grad()
            l = lossf(model(Xtr_t[b]), ytr_t[b])
            l.backward()
            if clip:
                nn.utils.clip_grad_norm_(model.parameters(), clip)
            opt.step()
            tot += l.detach().double() * len(b)
        model.eval()
        with torch.no_grad():
            vl = lossf(model(Xva_t), yva_t).item()
        hist.append((tot.item() / len(perm), vl))
        if verbose:
            print(ep, hist[-1])
        if vl < best - 1e-6:
            best, best_state, wait = vl, copy.deepcopy(model.state_dict()), 0
        else:
            wait += 1
            if wait >= patience:
                break
    model.load_state_dict(best_state)
    return model, np.array(hist)


def predict(model: nn.Module, X, batch: int = 4096) -> np.ndarray:
    model.eval()
    dev, out = _device_of(model), []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            out.append(model(torch.as_tensor(X[i:i + batch], device=dev)).cpu().numpy())
    return np.concatenate(out)


def recon_error(model: LSTMAutoencoder, X) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        Xt = torch.as_tensor(X, device=_device_of(model))
        return ((model(Xt) - Xt) ** 2).mean(dim=(1, 2)).cpu().numpy()


def diebold_mariano(e1: np.ndarray, e2: np.ndarray, h: int = 1) -> tuple[float, float]:
    """DM test of equal squared-error loss; HAC variance with h-1 lags.

    Returns (statistic, two-sided p-value); a negative statistic means that
    forecast 1 has the smaller loss.
    """
    from scipy import stats
    d = e1 ** 2 - e2 ** 2
    n = len(d); dbar = d.mean()
    gam = [np.sum((d[k:] - dbar) * (d[:n - k] - dbar)) / n for k in range(h)]
    var = gam[0] + 2 * sum(gam[1:])
    stat = dbar / np.sqrt(var / n)
    return float(stat), float(2 * stats.norm.sf(abs(stat)))
