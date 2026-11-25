"""Tests of the mathematical chapter (chapter 2 of Part 1) (synthetic data, offline): the identities and theorems the numerical examples rely on."""
import numpy as np
import pandas as pd
import pytest

from math_foundations.m0_boxes import exact_shapley
from math_foundations.m3_semirings import min_plus, segment_min_plus
from math_foundations.m4_lattices import HAN, closure_stats, ultrametric_stats
from math_foundations.m5_fuzzy_algebra import grid_checks
from math_foundations.m6_invariance import attention, position_codes, pvalues
from math_foundations.m7_index_numbers import carli, elementary_tests
from math_foundations.m8_algebraic_statistics import chisq, markov_chain_p
from math_foundations.m9_fixed_points import ga_chain, stationary


def test_min_plus_product_and_segmentation():
    v, M = np.array([0.0, 2.0]), np.array([[1.0, 5.0], [0.5, np.inf]])
    assert min_plus(v, M)[0].tolist() == [1.0, 5.0]           # min(0+1, 2+0.5) and min(0+5, 2+inf)
    from lab10_economies.patterns import segments
    rng = np.random.default_rng(0)
    y = np.r_[np.full(15, 1.0), np.full(20, 6.0), np.full(15, 3.0)] + rng.normal(0, 0.2, 50)
    cuts = segment_min_plus(y)[0]
    assert cuts == segments(y) and {15, 35} <= set(cuts)           # BIC may add short segments; the true cuts stay


def test_closed_itemsets_of_the_textbook_example():
    from lab5_fuzzy.apriori import apriori
    items = ["I1", "I2", "I3", "I4", "I5"]
    B = pd.DataFrame([[i in t for i in items] for t in HAN], columns=items)
    st = closure_stats(B, apriori(B, 2 / 9, max_len=5)[0])
    assert (st["frequent"], st["closed"], st["maximal"]) == (13, 9, 3)
    assert st["extensive"] == st["idempotent"] == st["monotone"] == st["galois"] == st["antitone"] == 1


def test_cophenetic_distance_is_an_ultrametric():
    rng = np.random.default_rng(1)
    P = rng.normal(size=(15, 3))
    D = np.sqrt(((P[:, None] - P[None]) ** 2).sum(2))
    st = ultrametric_stats(D)
    assert st["ultra_worst"] <= 1e-12 and st["same_cut"] == 1 and st["corr_worst"] > 0


def test_t_norm_laws_on_a_grid():
    g = grid_checks(21)
    assert g["order_holds"] == 1 and all(g[f"residuation_{k}"] == 1 for k in ("minimum", "product", "lukasiewicz"))
    assert g["mv_identity_err"] < 1e-12


def test_attention_is_permutation_equivariant_and_shifts_rotate_codes():
    rng = np.random.default_rng(2)
    X, W = rng.normal(size=(6, 4)), [rng.normal(size=(4, 4)) for _ in range(3)]
    Pm = np.eye(6)[rng.permutation(6)]
    assert np.allclose(attention(Pm @ X, *W), Pm @ attention(X, *W))
    PE = position_codes(6, 2)
    assert np.allclose(np.roll(PE, -1, axis=0) @ np.linalg.pinv(PE), np.roll(PE, -2, axis=0) @ np.linalg.pinv(np.roll(PE, -1, axis=0)))


def test_permutation_p_values_are_valid_probabilities():
    rng = np.random.default_rng(3)
    p_plain, p_cyclic = pvalues(rng.normal(size=40), rng.normal(size=40), rng, n_perm=49)
    assert 0 < p_plain <= 1 and 0 < p_cyclic <= 1


def test_elementary_index_axioms():
    P = np.array([[1.0, 2.0, 4.0], [1.2, 2.2, 3.0], [1.1, 2.6, 5.0], [1.5, 2.0, 4.4]])
    t = elementary_tests(P)
    assert t["jevons_reversal_max"] < 1e-12 and abs(t["jevons_chain_drift_pct"]) < 1e-10
    assert t["dutot_reversal_max"] < 1e-12 and abs(t["dutot_chain_drift_pct"]) < 1e-10
    assert carli(P[0], P[1]) * carli(P[1], P[0]) > 1                  # Carli fails time reversal (upward)


def test_markov_basis_walk_keeps_margins_and_gives_a_p_value():
    T = np.array([[5, 1], [1, 5]])
    p, acc = markov_chain_p(T, steps=4000, burn=200)
    assert 0 <= p <= 1 and acc > 0 and chisq(T) > 0


def test_genetic_algorithm_chain():
    P0, states = ga_chain(0.0)
    assert np.allclose(P0.sum(1), 1) and sum(P0[i, i] == 1 for i in range(len(states))) == 4   # absorbing populations
    P1, _ = ga_chain(0.1)
    pi = stationary(P1)
    assert np.allclose(pi @ P1, pi) and np.all(pi > 0)


def test_exact_shapley_values_of_a_linear_function():
    rng = np.random.default_rng(4)
    bg, x, beta = rng.normal(size=(50, 4)), rng.normal(size=4), np.array([1.0, -2.0, 0.0, 0.5])
    phi, v0, vN = exact_shapley(lambda Z: Z @ beta, x, bg)
    assert np.allclose(phi, beta * (x - bg.mean(0))) and phi.sum() == pytest.approx(vN - v0)
    assert phi[2] == 0                                                # a dummy gets nothing


def test_correlation_is_a_cosine_and_backpropagation_is_the_chain_rule():
    import torch
    from math_foundations.la_basics import backprop_two_layer, corr_as_cosine
    rng = np.random.default_rng(0)
    R = rng.normal(size=(200, 5)) @ rng.normal(size=(5, 5))
    assert np.allclose(corr_as_cosine(R), np.corrcoef(R.T))
    w, S = np.full(5, 0.2), np.cov(R.T)                             # portfolio variance is a quadratic form
    assert np.isclose(w @ S @ w, np.var(R @ w, ddof=1))
    X, y = rng.normal(size=(32, 4)), rng.integers(0, 2, 32).astype(float)
    W1, b1, w2, b2 = rng.normal(size=(3, 4)), rng.normal(size=3), rng.normal(size=3), 0.2
    loss, g = backprop_two_layer(X, y, W1, b1, w2, b2)
    T = {k: torch.tensor(v, requires_grad=True) for k, v in (("W1", W1), ("b1", b1), ("w2", w2), ("b2", np.array(b2)))}
    s = torch.relu(torch.tensor(X) @ T["W1"].T + T["b1"]) @ T["w2"] + T["b2"]
    lt = torch.nn.functional.binary_cross_entropy_with_logits(s, torch.tensor(y))
    lt.backward()
    assert np.isclose(loss, lt.item())
    assert all(np.allclose(g[k], T[k].grad.numpy()) for k in g)
