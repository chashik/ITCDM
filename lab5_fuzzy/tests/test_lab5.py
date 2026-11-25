"""Lab 5 tests: membership functions, FCM, Mamdani inference, Wang-Mendel rules."""
import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_blobs

from lab5_fuzzy.fuzzy import (Mamdani, Partition, WangMendel, fcm, partition_coefficient, partition_entropy,
                              trapmf, trimf)
from lab5_fuzzy.run import stress_system


def test_membership_functions():
    assert trimf(5, 0, 5, 10) == 1 and trimf(2.5, 0, 5, 10) == 0.5 and trimf(11, 0, 5, 10) == 0
    assert trapmf(3, 0, 2, 4, 6) == 1 and trapmf(5, 0, 2, 4, 6) == 0.5
    p = Partition(0, 1, 3)
    mu = p(np.linspace(-1, 4, 51))
    assert np.allclose(mu["low"] + mu["medium"] + mu["high"], 1)   # Ruspini partition


def test_fcm_memberships_and_crisp_limit():
    X, y = make_blobs(300, centers=3, cluster_std=0.4, random_state=0)
    U, V, J, it = fcm(X, 3, m=2.0)
    assert np.allclose(U.sum(1), 1) and (U >= 0).all()
    U2, *_ = fcm(X, 3, m=1.05)
    assert partition_coefficient(U2) > partition_coefficient(U) > 1 / 3
    assert partition_entropy(U2) < partition_entropy(U)
    from sklearn.metrics import adjusted_rand_score
    assert adjusted_rand_score(y, U.argmax(1)) > 0.95


def test_mamdani_single_rule_centroid():
    fis = Mamdani({"x": Partition(0, 5, 10)}, {"out": lambda u: trimf(u, 20, 50, 80)},
                  np.linspace(0, 100, 1001), [({"x": "high"}, "out")])
    assert fis({"x": np.array([10.0])})[0] == pytest.approx(50, abs=0.1)   # symmetric output set
    assert np.isnan(fis({"x": np.array([0.0])})[0])                      # no rule fires


def test_stress_index_is_monotone_in_vix():
    tr = pd.DataFrame({"vix": np.linspace(10, 45, 200), "credit_baa_aaa": np.linspace(0.6, 1.8, 200),
                       "ret_22": np.linspace(-0.1, 0.08, 200)})
    fis, rules = stress_system(tr)
    v = np.linspace(10, 50, 30)
    s = fis({"vix": v, "spread": np.full(30, 1.0), "mom": np.zeros(30)})
    assert np.all(np.diff(s) >= -1e-6) and s[-1] > s[0] + 30
    assert not np.isnan(fis({"vix": np.array([12.0]), "spread": np.array([1.0]), "mom": np.array([-0.05])})).any()


def test_wang_mendel_learns_threshold_rule():
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"a": rng.uniform(0, 10, 2000), "b": rng.uniform(0, 10, 2000)})
    y = (X["a"] > 7).astype(int).to_numpy()
    wm = WangMendel({"a": Partition(0, 5, 10), "b": Partition(0, 5, 10)}).fit(X, y)
    p = wm.predict_proba(pd.DataFrame({"a": [9.5, 1.0], "b": [5.0, 5.0]}))[:, 1]
    assert p[0] > 0.7 and p[1] < 0.1
    assert len(wm.rules_) <= 9
