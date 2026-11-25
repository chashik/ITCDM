"""Lab 3 (iteration 2). A multilayer perceptron for tabular data (PyTorch).

``fit_mlp`` trains a fully connected network with ReLU units, optional dropout,
AdamW or SGD, early stopping on the validation log-loss and returns the model
with its learning curves.  The seed is set before the network is constructed,
so a run is reproducible on the same hardware.
"""
from __future__ import annotations

import copy

import numpy as np
import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self, n_in: int, width: int = 64, depth: int = 2, dropout: float = 0.0):
        super().__init__()
        layers, d = [], n_in
        for _ in range(depth):
            layers += [nn.Linear(d, width), nn.ReLU()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            d = width
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def fit_mlp(Xtr, ytr, Xva, yva, width=64, depth=2, dropout=0.0, weight_decay=0.0, lr=1e-3, optimizer="adamw",
            batch=256, max_epochs=200, patience=10, early_stopping=True, seed=0):
    """Returns (model, history array of (train loss, validation loss) per epoch)."""
    torch.manual_seed(seed); np.random.seed(seed)
    model = MLP(Xtr.shape[1], width, depth, dropout)
    if optimizer == "sgd":
        opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    lossf = nn.BCEWithLogitsLoss()
    Xt, yt = torch.as_tensor(Xtr, dtype=torch.float32), torch.as_tensor(ytr, dtype=torch.float32)
    Xv, yv = torch.as_tensor(Xva, dtype=torch.float32), torch.as_tensor(yva, dtype=torch.float32)
    g = torch.Generator().manual_seed(seed)
    best, best_state, wait, hist = np.inf, None, 0, []
    for ep in range(max_epochs):
        model.train()
        perm = torch.randperm(len(Xt), generator=g)
        tot = 0.0
        for i in range(0, len(perm), batch):
            b = perm[i:i + batch]
            opt.zero_grad()
            l = lossf(model(Xt[b]), yt[b])
            if not torch.isfinite(l):
                break
            l.backward(); opt.step()
            tot += l.item() * len(b)
        model.eval()
        with torch.no_grad():
            vl = lossf(model(Xv), yv).item()
            tl = lossf(model(Xt), yt).item()
        hist.append((tl, vl))
        if not np.isfinite(vl):
            break
        if vl < best - 1e-6:
            best, best_state, wait = vl, copy.deepcopy(model.state_dict()), 0
        else:
            wait += 1
            if early_stopping and wait >= patience:
                break
    if early_stopping and best_state is not None:
        model.load_state_dict(best_state)
    return model, np.array(hist)


def predict_proba(model: nn.Module, X) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(torch.as_tensor(X, dtype=torch.float32))).numpy()
