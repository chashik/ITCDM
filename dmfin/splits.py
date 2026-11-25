"""Leakage-free splitting of time-ordered and cross-sectional data.

Two schemes are used in the paper.

1. Chronological hold-out (time series):  train | validation | test.
   A sample dated t whose target looks h days ahead covers t+1..t+h.  If
   t+h crosses the boundary of its block, the target overlaps the next
   block, so such samples are *purged* (dropped).

2. Expanding-window cross-validation with an embargo (``gap``) for model
   selection inside the training block: fold k trains on the first part
   of the data and validates on the next contiguous part, leaving ``gap``
   observations between them.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config as C


@dataclass
class ChronoSplit:
    train: pd.DatetimeIndex
    val: pd.DatetimeIndex
    test: pd.DatetimeIndex


def chrono_split(index: pd.DatetimeIndex, horizon: int = 0,
                 train_end: str = C.TRAIN_END, val_end: str = C.VAL_END,
                 test_end: str = C.TEST_END) -> ChronoSplit:
    """Split dates into blocks, purging the last ``horizon`` dates of each block."""
    idx = pd.DatetimeIndex(index).sort_values()
    blocks = []
    lo = idx.min() - pd.Timedelta(days=1)
    for hi in (pd.Timestamp(train_end), pd.Timestamp(val_end), pd.Timestamp(test_end)):
        b = idx[(idx > lo) & (idx <= hi)]
        if horizon > 0:
            b = b[:-horizon] if len(b) > horizon else b[:0]
        blocks.append(b)
        lo = hi
    return ChronoSplit(*blocks)


def expanding_window_cv(n: int, n_splits: int = 5, gap: int = 0,
                        min_train_frac: float = 0.4):
    """Yield (train_idx, val_idx) integer arrays for expanding-window CV.

    The first ``min_train_frac`` of the data is always in the training set;
    the remainder is cut into ``n_splits`` consecutive validation folds.
    ``gap`` observations before every validation fold are removed from
    training (embargo), which prevents overlapping-target leakage.
    """
    start = int(n * min_train_frac)
    edges = np.linspace(start, n, n_splits + 1).astype(int)
    for k in range(n_splits):
        v0, v1 = edges[k], edges[k + 1]
        tr = np.arange(0, max(0, v0 - gap))
        va = np.arange(v0, v1)
        yield tr, va


def make_windows(values: np.ndarray, targets: np.ndarray, positions: np.ndarray,
                 lookback: int):
    """Sliding windows for sequence models.

    For every position t in ``positions`` the input is values[t-lookback+1 .. t]
    and the label is targets[t].  Positions with insufficient history are
    skipped.  Inputs may reach back into an earlier block (they are past,
    observable information); labels never leave the block of ``positions``.
    """
    positions = positions[positions >= lookback - 1]
    X = np.stack([values[t - lookback + 1:t + 1] for t in positions]).astype(np.float32)
    y = targets[positions].astype(np.float32)
    return X, y, positions
