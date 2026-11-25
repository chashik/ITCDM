"""Worked examples of Part 3, chapter 4 (further methods): every number of the examples is computed here.

1. k-medoids: the mean and the medoid of five values with one outlier; PAM on seven values (cost after BUILD and after SWAP).
2. Spectral clustering: two concentric rings, k-means against normalized spectral clustering; the leading eigenvalues.
3. Hidden Markov model: two states of inflation ("calm", "stress"), forward likelihood, Viterbi path and posteriors of a
   short sequence.
4. FP-growth: the nine transactions of Han et al. (Table 6.1): FP-tree nodes, the conditional pattern base of I5, the
   frequent itemsets; Apriori's candidates on the same data.
5. Conditional inference: the quadratic statistic of a correlation, its p-value and the Bonferroni adjustment.
6. Sequential covering: FOIL gain of two candidate conditions.
7. Bayesian network: dollar depreciation and a food price shock as causes of rising inflation; explaining away.
Outputs: paper/generated/further_numbers.tex (macros \\Further...), paper/figures/further_rings.pdf.
Run (from code/):  python -m further_methods.examples
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import chi2
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

from dmfin import plotting as P
from dmfin.reporting import save_numbers
from lab5_fuzzy.apriori import apriori

from . import bayesnet, covering, fpgrowth, hmm, kmedoids, spectral


def main():
    num = {}
    # 1. k-medoids ------------------------------------------------------------------------------------------------
    v = np.array([1.0, 2.0, 3.0, 4.0, 40.0])
    D1 = np.abs(v[:, None] - v[None, :])
    med, _, _, _ = kmedoids.pam(D1, 1)
    num.update(kmMean=str(int(v.mean())), kmMedoid=str(int(v[med[0]])))
    pts = np.array([0, 1, 2, 5, 9, 10, 11], float)                # seven values on a line, k = 2
    D = np.abs(pts[:, None] - pts[None, :])
    n = len(pts)
    first = int(np.argmin(D.sum(axis=1)))
    near = D[:, [first]].min(axis=1)
    second = int(np.argmax([np.maximum(near - D[:, j], 0).sum() if j != first else -1 for j in range(n)]))
    build_cost = kmedoids.cost(D, [first, second])
    m, lab, c, swaps = kmedoids.pam(D, 2)
    num.update(kmBuildFirst=str(int(pts[first])), kmBuildSecond=str(int(pts[second])), kmBuildCost=int(build_cost),
               kmPamCost=int(c), kmSwaps=swaps, kmMedoids=" and ".join(str(int(pts[i])) for i in sorted(m)))
    print("k-medoids:", num)

    # 2. spectral clustering --------------------------------------------------------------------------------------
    rng = np.random.default_rng(1)
    t = rng.uniform(0, 2 * np.pi, 300)
    r = np.r_[np.full(150, 1.0), np.full(150, 3.0)] + rng.normal(0, 0.1, 300)
    X = np.c_[r * np.cos(t), r * np.sin(t)]
    y = np.repeat([0, 1], 150)
    W = spectral.gaussian_affinity(X, 0.3)
    lab_s, vals = spectral.njw(W, 2)
    lab_k = KMeans(2, n_init=10, random_state=0).fit_predict(X)
    num.update(spRandKmeans=adjusted_rand_score(y, lab_k), spRandSpectral=adjusted_rand_score(y, lab_s),
               spEigOne=float(vals[0]), spEigTwo=float(vals[1]), spEigThree=float(vals[2]), spSigma="0.3", spN=300)
    f, axs = P.fig(2.7, ncols=2)
    for ax, lb, ttl in ((axs[0], lab_k, "k-means"), (axs[1], lab_s, "spectral clustering")):
        for g, col in zip((0, 1), (P.BLUE, P.ORANGE)):
            ax.scatter(X[lb == g, 0], X[lb == g, 1], s=6, color=col)
        ax.set_aspect("equal"); ax.set_title(ttl, fontsize=9.5); ax.set_xticks([]); ax.set_yticks([])
    f.tight_layout(); P.save(f, "further_rings")
    print("spectral:", {k: v for k, v in num.items() if k.startswith("sp")})

    # 3. hidden Markov model --------------------------------------------------------------------------------------
    pi = np.array([0.8, 0.2])
    A = np.array([[0.9, 0.1], [0.2, 0.8]])
    mu, var = np.array([[3.0], [10.0]]), np.array([[1.5 ** 2], [3.0 ** 2]])
    x = np.array([[3.2], [4.1], [7.5], [9.8], [6.0]])
    ll, gamma, _ = hmm.forward_backward(x, pi, A, mu, var)
    path = hmm.viterbi(x, pi, A, mu, var)
    num.update(hmLogLik=ll, hmPath=", ".join("calm" if s == 0 else "stress" for s in path),
               hmPostThree=float(gamma[2, 1]), hmPostFive=float(gamma[4, 1]),
               hmDurCalm=float(1 / (1 - A[0, 0])), hmDurStress=float(1 / (1 - A[1, 1])))
    # the forward variables of the first two months, unscaled (for the hand calculation)
    B = hmm._emission(x, mu, var)
    a1 = pi * B[0]
    a2 = (a1 @ A) * B[1]
    num.update(hmBOneCalm=float(B[0, 0]), hmBOneStress=float(B[0, 1]), hmAOneCalm=float(a1[0]), hmAOneStress=float(a1[1]),
               hmATwoCalm=float(a2[0]), hmATwoStress=float(a2[1]))
    print("hmm:", {k: v for k, v in num.items() if k.startswith("hm")})

    # 4. FP-growth on Han et al., Table 6.1 -----------------------------------------------------------------------
    T = [["I1", "I2", "I5"], ["I2", "I4"], ["I2", "I3"], ["I1", "I2", "I4"], ["I1", "I3"], ["I2", "I3"], ["I1", "I3"],
         ["I1", "I2", "I3", "I5"], ["I1", "I2", "I3"]]
    items = sorted({i for t_ in T for i in t_})
    Bt = pd.DataFrame([[i in t_ for i in items] for t_ in T], columns=items)
    ff, st = fpgrowth.fpgrowth(Bt, 2 / 9)
    fa, lv = apriori(Bt, 2 / 9, max_len=5)
    assert ff.keys() == fa.keys()
    num.update(fpNodes=st["tree nodes"], fpCond=st["conditional trees"], fpFrequent=len(ff),
               fpAprioriCandidates=int(lv["candidates"].sum()), fpAprioriPasses=int((lv["candidates"] > 0).sum()))
    print("fp-growth:", {k: v for k, v in num.items() if k.startswith("fp")})

    # 5. conditional inference: c = (n - 1) r^2 ---------------------------------------------------------------------
    n_, r_, m_ = 20, 0.6, 3
    c_ = (n_ - 1) * r_ ** 2
    p_ = float(chi2.sf(c_, 1))
    num.update(ctN=n_, ctR=str(r_), ctStat=c_, ctP=p_, ctM=m_, ctPadj=min(1.0, m_ * p_))
    print("ctree:", {k: v for k, v in num.items() if k.startswith("ct")})

    # 6. FOIL gain -----------------------------------------------------------------------------------------------
    num.update(fgA=covering.foil_gain(10, 10, 8, 2), fgB=covering.foil_gain(10, 10, 4, 0))
    print("foil:", num["fgA"], num["fgB"])

    # 7. Bayesian network: explaining away -----------------------------------------------------------------------------
    cpt = {"D": pd.DataFrame([[0.7, 0.3]], columns=["no", "yes"]),
           "F": pd.DataFrame([[0.75, 0.25]], columns=["no", "yes"]),
           "I": pd.DataFrame([[0.2, 0.8], [0.4, 0.6], [0.5, 0.5], [0.85, 0.15]],
                             index=pd.MultiIndex.from_tuples([("yes", "yes"), ("yes", "no"), ("no", "yes"), ("no", "no")],
                                                             names=["D", "F"]), columns=["no", "yes"])}
    par = {"D": (), "F": (), "I": ("D", "F")}
    J = bayesnet.joint(cpt, par)
    num.update(bnPI=float(bayesnet.query(J, "I", {})["yes"]), bnPDgivenI=float(bayesnet.query(J, "D", {"I": "yes"})["yes"]),
               bnPDgivenIF=float(bayesnet.query(J, "D", {"I": "yes", "F": "yes"})["yes"]),
               bnPIgivenD=float(bayesnet.query(J, "I", {"D": "yes"})["yes"]))
    print("bn:", {k: v for k, v in num.items() if k.startswith("bn")})
    save_numbers("further", num)


if __name__ == "__main__":
    main()
