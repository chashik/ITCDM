"""Lab 5 (iteration 2). Frequent itemsets and association rules (Apriori).

A transaction is a set of items such as ``pay_1=late``.  ``apriori`` finds all
itemsets whose support is at least ``min_support``, level by level: candidate
itemsets of size k+1 are joined from frequent itemsets of size k and pruned if any
of their k-subsets is infrequent (the Apriori property).  ``rules`` derives rules
X -> Y with support, confidence and lift.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd


def one_hot(items: pd.DataFrame) -> pd.DataFrame:
    """Columns of categorical attributes -> boolean item matrix with columns 'attribute=value'."""
    return pd.concat([(items[c].astype(str).radd(f"{c}=")).str.get_dummies().astype(bool) for c in items], axis=1)


def apriori(B: pd.DataFrame, min_support: float, max_len: int = 4):
    """Returns (frequent itemsets {frozenset: support}, statistics per level)."""
    M = B.to_numpy(); names = list(B.columns); n = len(M)
    col = {name: M[:, j] for j, name in enumerate(names)}
    sup = lambda s: float(np.logical_and.reduce([col[i] for i in s]).sum()) / n
    level = {frozenset([i]): sup([i]) for i in names}
    level = {k: v for k, v in level.items() if v >= min_support}
    freq, stats = dict(level), [(1, len(names), len(level))]
    k = 1
    while level and k < max_len:
        prev = sorted(level, key=lambda s: sorted(s))
        cands = set()
        for a, b in combinations(prev, 2):
            u = a | b
            attrs = [i.split("=")[0] for i in u]
            if len(u) == k + 1 and len(set(attrs)) == len(attrs) and all(frozenset(s) in level for s in combinations(u, k)):
                cands.add(u)
        level = {c: sup(c) for c in cands}
        n_cand = len(level)
        level = {c: v for c, v in level.items() if v >= min_support}
        freq.update(level)
        stats.append((k + 1, n_cand, len(level)))
        k += 1
    return freq, pd.DataFrame(stats, columns=["size", "candidates", "frequent"])


def rules(freq: dict, consequents: list[str], min_conf: float = 0.0) -> pd.DataFrame:
    """Rules X -> {c} for the given consequent items."""
    out = []
    for s, sup_xy in freq.items():
        for c in consequents:
            if c in s and len(s) > 1:
                x = s - {c}
                sx, sy = freq.get(x), freq.get(frozenset([c]))
                if sx and sy:
                    conf = sup_xy / sx
                    if conf >= min_conf:
                        out.append((" AND ".join(sorted(x)), c, sup_xy, conf, conf / sy, len(x)))
    return pd.DataFrame(out, columns=["antecedent", "consequent", "support", "confidence", "lift", "length"])
