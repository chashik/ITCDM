"""Rule induction by sequential covering (Han et al., p. 359-363).

Rules for one class are learned one at a time. A rule grows from the empty condition by adding, each time, the
condition "attribute <= c" or "attribute > c" with the largest FOIL gain
    FOIL_Gain = pos' (log2(pos' / (pos' + neg')) - log2(pos / (pos + neg))),
pos and neg (pos', neg') being the positive and negative examples covered before (after) the condition is added; growth
stops when no condition has a positive gain or the rule covers no negative example. The examples covered by the rule
are then removed (p. 359) and the next rule is learned, until no rule with the required accuracy and coverage is found.
"""
from __future__ import annotations

import numpy as np


def foil_gain(pos: int, neg: int, pos1: int, neg1: int) -> float:
    if pos1 == 0:
        return -np.inf
    return pos1 * (np.log2(pos1 / (pos1 + neg1)) - np.log2(pos / (pos + neg)))


class SequentialCovering:
    def __init__(self, max_cuts: int = 200, min_pos: int = 5, min_accuracy: float = 0.6, max_rules: int = 6, max_conds: int = 3):
        self.max_cuts, self.min_pos, self.min_accuracy, self.max_rules, self.max_conds = max_cuts, min_pos, min_accuracy, max_rules, max_conds

    def _cuts(self, X):
        """Midpoints between consecutive distinct values (at most max_cuts per attribute, evenly spaced)."""
        out = []
        for j in range(X.shape[1]):
            u = np.unique(X[:, j])
            mid = (u[1:] + u[:-1]) / 2
            if len(mid) > self.max_cuts:
                mid = mid[np.linspace(0, len(mid) - 1, self.max_cuts).round().astype(int)]
            out.append(mid)
        return out

    def _learn_one(self, X, pos_mask, cuts):
        cover = np.ones(len(X), bool)
        conds = []
        while len(conds) < self.max_conds:
            pos, neg = int((cover & pos_mask).sum()), int((cover & ~pos_mask).sum())
            if neg == 0:
                break
            best = (0.0, None)
            for j, cs in enumerate(cuts):
                for c in cs:
                    for op in ("<=", ">"):
                        m = cover & ((X[:, j] <= c) if op == "<=" else (X[:, j] > c))
                        g = foil_gain(pos, neg, int((m & pos_mask).sum()), int((m & ~pos_mask).sum()))
                        if g > best[0] + 1e-12:
                            best = (g, (j, op, float(c)))
            if best[1] is None:
                break
            j, op, c = best[1]
            conds.append(best[1])
            cover &= (X[:, j] <= c) if op == "<=" else (X[:, j] > c)
        return conds, cover

    def fit(self, X, y, target, names=None) -> "SequentialCovering":
        X, y = np.asarray(X, float), np.asarray(y)
        self.names = list(names) if names is not None else [f"x{j}" for j in range(X.shape[1])]
        cuts = self._cuts(X)
        pos_left = y == target
        remaining = np.ones(len(X), bool)                        # examples not yet covered by a rule
        self.rules_ = []
        while len(self.rules_) < self.max_rules and pos_left[remaining].sum() >= self.min_pos:
            conds, cover = self._learn_one(X[remaining], pos_left[remaining], cuts)
            if not conds:
                break
            idx = np.flatnonzero(remaining)[cover]
            p, nn = int(pos_left[idx].sum()), int((~pos_left[idx]).sum())
            if p < self.min_pos or p / (p + nn) < self.min_accuracy:
                break
            self.rules_.append({"conditions": conds, "positives": p, "negatives": nn})
            remaining[idx] = False
        self.target = target
        return self

    def covers(self, X) -> np.ndarray:
        X = np.asarray(X, float)
        out = np.zeros(len(X), bool)
        for r in self.rules_:
            m = np.ones(len(X), bool)
            for j, op, c in r["conditions"]:
                m &= (X[:, j] <= c) if op == "<=" else (X[:, j] > c)
            out |= m
        return out

    def text(self, fmt=lambda v: f"{v:.1f}") -> list[str]:
        return [" AND ".join(f"{self.names[j]} {op} {fmt(c)}" for j, op, c in r["conditions"]) for r in self.rules_]
