"""Tests of the further methods (Part 3, chapter 4) on synthetic data with known answers."""
import itertools

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics import adjusted_rand_score

from further_methods import bayesnet, ctree, covering, fpgrowth, hmm, kmedoids, spectral
from lab5_fuzzy.apriori import apriori


def blobs(n_per=15, seed=0):
    rng = np.random.default_rng(seed)
    X = np.vstack([rng.normal(c, 0.4, (n_per, 2)) for c in ((0, 0), (4, 0), (2, 4))])
    return X, np.repeat([0, 1, 2], n_per)


def test_pam_finds_the_best_medoids_of_three_groups():
    X, _ = blobs(5)
    D = cdist(X, X)
    best = min(kmedoids.cost(D, m) for m in itertools.combinations(range(len(X)), 3))
    _, _, c, _ = kmedoids.pam(D, 3)
    assert abs(c - best) < 1e-9


def test_pamk_chooses_three_clusters_and_clara_approximates_pam():
    X, y = blobs(20)
    D = cdist(X, X)
    k, asw = kmedoids.pamk(D, range(2, 6))
    assert k == 3 and asw[3] > 0.7
    _, lab, c, _ = kmedoids.pam(D, 3)
    assert adjusted_rand_score(y, lab) == 1.0
    _, _, c_full, _ = kmedoids.clara(X, 3, n_samples=1, sample_size=len(X))
    assert abs(c_full - c) < 1e-9                                  # a sample of all objects is PAM itself
    _, _, c_small, trials = kmedoids.clara(X, 3, n_samples=5, sample_size=12)
    assert c_small >= c - 1e-9 and min(trials) == c_small


def test_spectral_clustering_separates_two_rings_where_kmeans_fails():
    rng = np.random.default_rng(1)
    t = rng.uniform(0, 2 * np.pi, 300)
    r = np.r_[np.full(150, 1.0), np.full(150, 3.0)] + rng.normal(0, 0.1, 300)
    X = np.c_[r * np.cos(t), r * np.sin(t)]
    y = np.repeat([0, 1], 150)
    W = spectral.gaussian_affinity(X, 0.3)
    lab, vals = spectral.njw(W, 2)
    assert adjusted_rand_score(y, lab) == 1.0
    assert adjusted_rand_score(y, KMeans(2, n_init=10, random_state=0).fit_predict(X)) < 0.1
    ref = SpectralClustering(2, affinity="precomputed", random_state=0).fit_predict(W)
    assert adjusted_rand_score(lab, ref) == 1.0
    Xb, _ = blobs(30)                                             # the eigengap heuristic needs compact groups
    assert spectral.eigengap(spectral.gaussian_affinity(Xb, 1.0)) == 3


def test_hmm_forward_algorithm_equals_the_sum_over_all_state_paths():
    pi, A = np.array([0.6, 0.4]), np.array([[0.7, 0.3], [0.2, 0.8]])
    mu, var = np.array([[0.0], [2.0]]), np.array([[1.0], [0.5]])
    X = np.array([[0.1], [1.9], [2.2], [-0.3]])
    B = hmm._emission(X, mu, var)
    total = sum(pi[q[0]] * B[0, q[0]] * np.prod([A[q[t - 1], q[t]] * B[t, q[t]] for t in range(1, 4)])
                for q in itertools.product(range(2), repeat=4))
    ll, gamma, _ = hmm.forward_backward(X, pi, A, mu, var)
    assert abs(ll - np.log(total)) < 1e-10
    assert np.allclose(gamma.sum(axis=1), 1)


def test_baum_welch_recovers_the_parameters_of_a_two_state_chain():
    rng = np.random.default_rng(2)
    A = np.array([[0.95, 0.05], [0.10, 0.90]])
    q = [0]
    for _ in range(1999):
        q.append(rng.choice(2, p=A[q[-1]]))
    q = np.array(q)
    x = np.where(q == 0, 0.0, 3.0) + rng.normal(0, 1, len(q))
    m = hmm.GaussianHMM(2, n_starts=3, seed=0).fit([x])
    assert np.allclose(m.mu[:, 0], [0, 3], atol=0.15)
    assert np.allclose(np.diag(m.A), [0.95, 0.90], atol=0.03)
    assert (m.decode(x) == q).mean() > 0.95


def test_fpgrowth_finds_the_same_itemsets_as_apriori():
    # Han et al., Table 6.1 (p. 250) with a minimum count of 2: 13 frequent itemsets, e.g. {I1, I2, I5}: 2
    T = [["I1", "I2", "I5"], ["I2", "I4"], ["I2", "I3"], ["I1", "I2", "I4"], ["I1", "I3"], ["I2", "I3"], ["I1", "I3"],
         ["I1", "I2", "I3", "I5"], ["I1", "I2", "I3"]]
    items = sorted({i for t in T for i in t})
    B = pd.DataFrame([[i in t for i in items] for t in T], columns=items)
    f, _ = fpgrowth.fpgrowth(B, 2 / 9)
    assert len(f) == 13 and abs(f[frozenset({"I1", "I2", "I5"})] - 2 / 9) < 1e-12
    rng = np.random.default_rng(3)
    base = rng.random((400, 1)) < 0.5
    R = pd.DataFrame(np.c_[base & (rng.random((400, 4)) < 0.8), rng.random((400, 6)) < 0.3],
                     columns=[f"i{j}" for j in range(10)])
    fa, _ = apriori(R, 0.08, max_len=4)
    ff, _ = fpgrowth.fpgrowth(R, 0.08, max_len=4)
    assert fa.keys() == ff.keys() and all(abs(fa[s] - ff[s]) < 1e-12 for s in fa)


def test_ctree_statistic_is_the_correlation_test_and_the_tree_finds_the_step():
    rng = np.random.default_rng(4)
    x, y = rng.normal(size=200), rng.normal(size=200)
    y = y + 0.3 * x
    c, df = ctree.linear_test(x[:, None], y[:, None])
    r = np.corrcoef(x, y)[0, 1]
    assert df == 1 and abs(c - (len(x) - 1) * r ** 2) < 1e-8      # c_quad = (n - 1) r^2
    X = rng.uniform(size=(300, 2))
    y = 2.0 * (X[:, 0] > 0.6) + rng.normal(0, 0.5, 300)
    t = ctree.CTree().fit(X, y, names=["x1", "noise"])
    assert t.root.var == 0 and abs(t.root.cut - 0.6) < 0.05
    assert "noise" not in t.splits()
    t0 = ctree.CTree().fit(X, rng.normal(size=300))               # no association: no split
    assert t0.root.var is None and t0.root.pvalue >= 0.05


def test_sequential_covering_learns_a_known_rule():
    rng = np.random.default_rng(5)
    X = rng.uniform(size=(600, 3))
    y = (X[:, 0] > 0.5) & (X[:, 1] <= 0.3)
    sc = covering.SequentialCovering(min_accuracy=0.9).fit(X, y, True, names=["a", "b", "c"])
    first = {(j, op) for j, op, _ in sc.rules_[0]["conditions"]}
    assert first == {(0, ">"), (1, "<=")}
    pred = sc.covers(X)
    assert (pred & y).sum() / y.sum() > 0.9 and (pred & y).sum() / pred.sum() > 0.9


def test_bayesian_network_recovers_a_v_structure_and_inference_is_bayes_rule():
    rng = np.random.default_rng(6)
    n = 4000
    a, b = rng.random(n) < 0.5, rng.random(n) < 0.4
    c = rng.random(n) < np.where(a & b, 0.9, np.where(a | b, 0.5, 0.1))
    D = pd.DataFrame({"A": a.astype(int), "B": b.astype(int), "C": c.astype(int)})
    parents, _, _ = bayesnet.hill_climb(D)
    assert set(parents["C"]) == {"A", "B"} and parents["A"] == () and parents["B"] == ()
    # inference by enumeration on A -> B with known tables: P(A=1 | B=1) by Bayes' rule
    cpt = {"A": pd.DataFrame([[0.7, 0.3]], columns=[0, 1]),
           "B": pd.DataFrame([[0.9, 0.1], [0.2, 0.8]], index=pd.Index([0, 1], name="A"), columns=[0, 1])}
    J = bayesnet.joint(cpt, {"A": (), "B": ("A",)})
    p = bayesnet.query(J, "A", {"B": 1})
    assert abs(p[1] - 0.3 * 0.8 / (0.3 * 0.8 + 0.7 * 0.1)) < 1e-12
