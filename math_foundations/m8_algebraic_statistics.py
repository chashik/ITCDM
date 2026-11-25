"""Part 1, chapter 2 (mathematical foundations), section M8. Algebraic statistics: an exact test of independence on the contingency table of Lab 1.

Lab 1 crosses the 11 sectors of the 110 stocks with their risk clusters.  Under independence with the margins fixed,
the table is uniformly distributed (hypergeometrically) over its fibre: all non-negative integer tables with the same
row and column sums.  The fibre is connected by the basic moves +1 -1 / -1 +1 on 2 x 2 sub-tables — a Markov basis,
the generators of the toric ideal of the independence model (Diaconis and Sturmfels 1998; Drton, Sturmfels, Sullivant
2009, ch. 1).  A Metropolis walk with these moves gives an exact conditional p-value of the chi-square statistic,
compared with the asymptotic chi-square p-value, which is unreliable for this sparse table (all expected counts < 5):
under independence (labels shuffled, margins kept) its rejection rate at the 5 % level is measured.
A 2 x 2 example shows the fibre and its single move.
Outputs: paper/generated/math8_numbers.tex.  Run (from code/):  python -m math_foundations.m8_algebraic_statistics
"""
from __future__ import annotations

from math import lgamma

import numpy as np
import pandas as pd
from scipy.stats import chi2, chi2_contingency

from dmfin.reporting import save_numbers


def chisq(T: np.ndarray) -> float:
    E = T.sum(1, keepdims=True) * T.sum(0, keepdims=True) / T.sum()
    return float(((T - E) ** 2 / E).sum())


def log_hyper(T: np.ndarray) -> float:
    """log of the probability of T in its fibre up to a constant: -sum log(t_ij!)."""
    return -sum(lgamma(v + 1) for v in T.ravel())


STEPS, BURN = 200000, 5000                                   # Metropolis steps; the first BURN are discarded


def markov_chain_p(T0: np.ndarray, steps: int = STEPS, burn: int = BURN, seed: int = 0) -> tuple[float, float]:
    """Metropolis walk on the fibre with the basic moves; returns (p-value, acceptance rate)."""
    rng = np.random.default_rng(seed)
    T, lp = T0.copy(), log_hyper(T0)
    s0, r, c = chisq(T0), *T0.shape
    hits = acc = kept = 0
    for it in range(steps):
        i, k = rng.choice(r, 2, replace=False)
        j, l = rng.choice(c, 2, replace=False)
        sgn = 1 if rng.random() < 0.5 else -1
        M = np.zeros_like(T)
        M[i, j] = M[k, l] = sgn; M[i, l] = M[k, j] = -sgn
        U = T + M
        if (U >= 0).all():
            lq = log_hyper(U)
            if np.log(rng.random()) < lq - lp:
                T, lp = U, lq
                acc += 1
        if it >= burn:
            kept += 1
            hits += chisq(T) >= s0 - 1e-9
    return hits / kept, acc / steps


def main():
    from dmfin import config as C
    cl = pd.read_csv(C.RESULTS_DIR / "lab1_clusters.csv")
    ct = pd.crosstab(cl["sector"], cl["risk_cluster"])
    T = ct.to_numpy()
    stat, p_asym, dof, E = chi2_contingency(T, correction=False)
    p_mc, acc = markov_chain_p(T)
    num = {"rows": T.shape[0], "cols": T.shape[1], "stocks": int(T.sum()), "chisq": stat, "dof": dof,
           "p_asymptotic": p_asym, "p_exact_mc": p_mc, "mc_kept": STEPS - BURN, "acceptance": 100 * acc,
           "small_expected_share": 100 * float((E < 5).mean()), "zero_cells": int((T == 0).sum())}
    # size of the asymptotic test on this sparse table: shuffled labels (independence, the same margins)
    rng = np.random.default_rng(1)
    pa = []
    for _ in range(1000):
        sh = pd.crosstab(cl["sector"].to_numpy(), rng.permutation(cl["risk_cluster"].to_numpy())).to_numpy()
        pa.append(chi2_contingency(sh, correction=False)[1])
    pa = np.array(pa)
    num.update({"null_tables": len(pa), "null_reject5_asym": 100 * float((pa <= 0.05).mean()),
                "null_reject1_asym": 100 * float((pa <= 0.01).mean())})
    # 2 x 2 example: margins (3, 2) x (2, 3) -> the fibre and the single move
    fibre = [np.array([[a, 3 - a], [2 - a, a]]) for a in range(0, 3) if 3 - a >= 0 and 2 - a >= 0]
    num.update({"fibre_2x2_size": len(fibre),
                "fibre_2x2_connected": int(all(np.abs(fibre[t + 1] - fibre[t]).sum() == 4 for t in range(len(fibre) - 1)))})
    save_numbers("math8", num)
    print(pd.Series(num).to_string())


if __name__ == "__main__":
    main()
