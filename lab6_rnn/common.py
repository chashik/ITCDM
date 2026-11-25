"""Shared data preparation for the Lab 6 scripts."""
from __future__ import annotations

import functools

import numpy as np
import torch

from dmfin.data import load_market_panel
from dmfin.features import add_targets, market_features

from .nets import build_inputs

torch.set_num_threads(2)


@functools.lru_cache(maxsize=1)
def lab6_frame():
    panel = load_market_panel()
    feats = add_targets(market_features(panel), horizons=(1, 5, 22))
    return build_inputs(feats, panel), panel


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def r2(y, p):
    y, p = np.asarray(y), np.asarray(p)
    return float(1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2))
