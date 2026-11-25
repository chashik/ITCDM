"""Lab 7 (iteration 2) tests: publication lags, the extended store, tables for any groups, agents."""
import numpy as np
import pandas as pd
import pytest

from lab7_rl_features import ext as X


def toy_tables(steps=20):
    """Two groups; only group 0 is useful (RMSE 0.5 alone, 0.9 together with group 1)."""
    rm = np.full((steps, 4), np.nan); rm[:, 1], rm[:, 2], rm[:, 3] = 0.5, 1.0, 0.9
    z = np.zeros(steps)
    return X.Tables(["a", "b"], pd.bdate_range("2020-01-01", periods=steps), rm, rm.copy(), rm.copy(), z,
                    np.full(steps, 3), np.full(steps, 20.0), z, z, np.full(steps, 21))


def test_align_respects_the_publication_lag():
    weekly = pd.Series([1.0, 2.0], index=pd.to_datetime(["2020-01-03", "2020-01-10"]))   # Fridays
    idx = pd.bdate_range("2020-01-13", "2020-01-20")
    a = X._align(weekly, idx, lag_days=7)
    assert (a.loc[:"2020-01-16"] == 1.0).all()          # the value of 10 January is not known yet
    assert a.loc["2020-01-17"] == 2.0


def test_extend_documents_keeps_old_fields():
    docs = [{"date": d.strftime("%Y-%m-%d"), "src": "feed-A", "fields": {"ret_5": 0.01}} for d in pd.bdate_range("2020-01-01", periods=400)]
    new = pd.DataFrame({"skew": 120.0}, index=pd.bdate_range("2020-01-01", periods=400))
    out = X.extend_documents(docs, new, seed=0)
    assert all(d["fields"]["ret_5"] == 0.01 for d in out) and "skew" not in docs[0]["fields"]
    share = np.mean(["skew" in d["fields"] for d in out])
    assert 0.95 < share < 1.0                             # outages with probability 2 %


def test_tables_supervised_and_unlabeled_criteria():
    n = 1000
    idx = pd.bdate_range("2014-01-01", periods=n)
    rng = np.random.default_rng(0)
    a = rng.normal(size=(n, 2))
    f = pd.DataFrame({"a1": a[:, 0], "a2": a[:, 1], "b1": a[:, 0] + 0.3 * rng.normal(size=n),
                      "c1": rng.normal(size=n), "log_vix": np.log(15 + 5 * rng.uniform(size=n))}, index=idx)
    f["y"] = 0.8 * f["a1"] + 0.1 * rng.normal(size=n)
    groups = {"a": ["a1", "a2"], "b": ["b1"], "c": ["c1", "log_vix"]}
    T = X.build_tables(f, groups, first=800, last=850, stress=pd.Series(0.0, index=idx), workers=2)
    k = 0
    assert T.avail[k] == 7 and T.nobs[k] == 21
    assert T.rmse[k, 1] < T.rmse[k, 2] < T.rmse[k, 4]        # a beats its noisy copy b, which beats c
    assert T.recon[k, 0b101] < 0.05                          # a and c reconstruct b: no labels needed
    assert T.recon[k, 0b011] > 0.3                            # without c, its variance is lost
    assert np.isfinite(T.lasso[k])


def test_reward_is_used_only_after_the_next_decision():
    T = toy_tables(3)
    ag = X.Tabular(3, alpha=1.0, gamma=0.0, eps=0.0)
    seen, act = [], ag.act

    def spy(env, s, greedy=False):
        seen.append(sum(float(np.abs(v).sum()) for v in ag.Q.values()))
        return act(env, s, greedy)
    ag.act = spy
    ag.episode(X.Env(T, np.arange(3)))
    assert seen[0] == 0 and seen[1] == 0 and seen[2] > 0


@pytest.mark.parametrize("make", [lambda: X.Tabular(3, alpha=0.3, gamma=0.5, seed=0),
                                  lambda: X.Tabular(3, alpha=0.3, gamma=0.5, sarsa=True, seed=0),
                                  lambda: X.LinUCB(3, 8, alpha=0.5),
                                  lambda: X.DQN(3, 8, lr=3e-3, seed=0, batch=32, sync=50)])
def test_agents_learn_to_drop_the_useless_group(make):
    T = toy_tables()
    ag = make()
    for ep in range(80):
        ag.eps = X.eps_schedule(ep, 80)
        ag.episode(X.Env(T, np.arange(20)))
    ag.eps = 0.0
    e, m, _ = ag.episode(X.Env(T, np.arange(20)), learn=False, greedy=True)
    assert np.mean(m == 1) > 0.9 and X.pooled(e, T.nobs) < 0.55


def test_reset_from_a_given_or_random_subset():
    T = toy_tables(3)
    T.avail[:] = 1                                         # only group 0 is available
    env = X.Env(T, np.arange(3))
    env.reset(mask=2)
    assert env.mask == 1                                   # an unavailable start falls back to the available groups
    rng = np.random.default_rng(0)
    assert all(env.random_mask(rng) == 1 for _ in range(20))
