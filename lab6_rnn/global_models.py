"""Lab 6 (iteration 2). Sequence models trained on many series at once.

The panel of ``dmfin.panel`` holds 62 daily series.  A window is the last ``L``
rows of one series; windows are gathered by index arithmetic, so that the whole
panel stays in (GPU) memory once and no window is materialized in advance.

Models: LSTM, GRU, a temporal convolutional network (dilated causal
convolutions) and a Transformer encoder; all map a window to the log realized
volatility of the next five days of the same series.
"""
from __future__ import annotations

import copy
import math

import numpy as np
import torch
from torch import nn


class RNNHead(nn.Module):
    def __init__(self, n_in, hidden=64, cell="lstm"):
        super().__init__()
        self.rnn = {"lstm": nn.LSTM, "gru": nn.GRU}[cell](n_in, hidden, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        h, _ = self.rnn(x)
        return self.head(h[:, -1]).squeeze(-1)


class CausalConv(nn.Module):
    def __init__(self, c_in, c_out, k, d):
        super().__init__()
        self.pad = (k - 1) * d
        self.conv = nn.Conv1d(c_in, c_out, k, dilation=d)

    def forward(self, x):                        # x: (batch, channels, time)
        return self.conv(nn.functional.pad(x, (self.pad, 0)))


class TCN(nn.Module):
    """Residual blocks of dilated causal convolutions (Bai, Kolter, Koltun 2018)."""

    def __init__(self, n_in, channels=32, kernel=3, dilations=(1, 2, 4, 8), dropout=0.1):
        super().__init__()
        self.inp = nn.Conv1d(n_in, channels, 1)
        self.blocks = nn.ModuleList([nn.Sequential(CausalConv(channels, channels, kernel, d), nn.ReLU(), nn.Dropout(dropout),
                                                   CausalConv(channels, channels, kernel, d), nn.ReLU(), nn.Dropout(dropout))
                                     for d in dilations])
        self.head = nn.Linear(channels, 1)

    def forward(self, x):
        z = self.inp(x.transpose(1, 2))
        for b in self.blocks:
            z = z + b(z)
        return self.head(z[:, :, -1]).squeeze(-1)


class TransformerHead(nn.Module):
    """Transformer encoder with learned positional embeddings; the last position predicts."""

    def __init__(self, n_in, d_model=32, heads=4, layers=2, lookback=20, dropout=0.1):
        super().__init__()
        self.inp = nn.Linear(n_in, d_model)
        self.pos = nn.Parameter(torch.zeros(1, lookback, d_model))
        layer = nn.TransformerEncoderLayer(d_model, heads, 4 * d_model, dropout, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, layers)
        self.head = nn.Linear(d_model, 1)
        mask = torch.triu(torch.full((lookback, lookback), float("-inf")), 1)      # causal attention
        self.register_buffer("mask", mask)

    def forward(self, x):
        z = self.inp(x) + self.pos[:, -x.shape[1]:]
        z = self.enc(z, mask=self.mask[-x.shape[1]:, -x.shape[1]:])
        return self.head(z[:, -1]).squeeze(-1)


def build(name, n_in, lookback):
    if name in ("lstm", "gru"):
        return RNNHead(n_in, 64, name)
    if name == "tcn":
        return TCN(n_in)
    if name == "transformer":
        return TransformerHead(n_in, lookback=lookback)
    raise ValueError(name)


def n_params(model) -> int:
    return sum(p.numel() for p in model.parameters())


def train_global(name, X, y, rows_tr, rows_va, lookback, seed=0, device="cpu", batch=1024, lr=1e-3,
                 max_epochs=30, patience=5):
    """X: (rows, features) float32 array of the whole panel; rows_*: row numbers that end a window."""
    torch.manual_seed(seed); np.random.seed(seed)
    model = build(name, X.shape[1], lookback).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    Xt = torch.as_tensor(X, device=device); yt = torch.as_tensor(y, device=device)
    offs = torch.arange(-lookback + 1, 1, device=device)
    rtr = torch.as_tensor(rows_tr, device=device); rva = torch.as_tensor(rows_va, device=device)
    win = lambda r: Xt[r[:, None] + offs[None, :]]
    lossf = nn.MSELoss()
    g = torch.Generator().manual_seed(seed)
    best, best_state, wait, hist = math.inf, None, 0, []
    for ep in range(max_epochs):
        model.train()
        perm = rtr[torch.randperm(len(rtr), generator=g).to(device)]
        tot = torch.zeros((), device=device, dtype=torch.float64)
        for i in range(0, len(perm), batch):
            r = perm[i:i + batch]
            opt.zero_grad()
            l = lossf(model(win(r)), yt[r])
            l.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += l.detach().double() * len(r)
        vl = mse(model, win, rva, yt)
        hist.append((tot.item() / len(perm), vl))
        if vl < best - 1e-6:
            best, best_state, wait = vl, copy.deepcopy(model.state_dict()), 0
        else:
            wait += 1
            if wait >= patience:
                break
    model.load_state_dict(best_state)
    return model, np.array(hist), win


@torch.no_grad()
def mse(model, win, rows, yt, batch=8192):
    model.eval()
    tot = 0.0
    for i in range(0, len(rows), batch):
        r = rows[i:i + batch]
        tot += float(((model(win(r)) - yt[r]) ** 2).sum())
    return tot / len(rows)


@torch.no_grad()
def predict_rows(model, win, rows, device, batch=8192):
    model.eval()
    r = torch.as_tensor(rows, device=device)
    return np.concatenate([model(win(r[i:i + batch])).float().cpu().numpy() for i in range(0, len(r), batch)])
