"""Lab 7 tests: NoSQL stream, schema normalisation, environment, Q-learning."""
import numpy as np
import pandas as pd
import pytest

from lab7_rl_features.env import (ALL, KEEP, NG, FeatureSelectionEnv, QAgent, Tables, build_tables,
                                  mask_features, popcount, regime, run_episode)
from lab7_rl_features.stream import FEATURES, MarketStore, canonical_frame, normalise, to_documents


def toy_frame(n=1300, start="2014-01-01"):
    idx = pd.bdate_range(start, periods=n)
    rng = np.random.default_rng(0)
    d = pd.DataFrame(rng.normal(size=(n, len(FEATURES))), index=idx, columns=FEATURES)
    d["log_vix"] = np.log(15 + 5 * rng.uniform(size=n))
    d["y"] = 0.8 * d["log_pk_m"] + 0.1 * rng.normal(size=n)
    return d


def test_schema_drift_is_normalised():
    f = toy_frame(30, start="2015-12-20")
    docs = to_documents(f, seed=1)
    late = [d for d in docs if d["date"] >= "2016-01-01"]
    assert all("log_vix" not in d["fields"] for d in late)
    assert any("VIX" in d["fields"] for d in late)
    n = normalise({"VIX": 20.0, "term_bp": 150.0})
    assert n["log_vix"] == pytest.approx(np.log(20)) and n["term_10y3m"] == pytest.approx(1.5)


def test_store_roundtrip_and_audit():
    f = toy_frame(60, start="2015-12-01")
    st = MarketStore(); st.ingest(to_documents(f, seed=2))
    w = st.window("2015-12-01", "2016-02-28")
    assert list(w.columns) == FEATURES + ["y"]
    assert w["log_vix"].notna().mean() > 0.9                      # restored across the drift
    assert np.allclose(w["log_vix"].dropna(), f.loc[w["log_vix"].dropna().index, "log_vix"])
    audit = st.schema_versions()
    assert audit.loc["VIX", "2016"] and not audit.loc["VIX", "2015"]


def test_tables_respect_label_delay_and_find_signal():
    f = toy_frame()
    st = MarketStore(); st.ingest(to_documents(f, seed=3))
    T = build_tables(st, f.index, first=800, last=900)
    vol = 1 << list(__import__("lab7_rl_features.stream", fromlist=["GROUPS"]).GROUPS).index("vol")
    k = 0
    others = [m for m in range(1, ALL + 1) if not m & vol and not np.isnan(T.rmse[k, m])]
    assert T.rmse[k, vol] < min(T.rmse[k, m] for m in others)   # the informative group wins
    assert T.nobs[0] == 21


def test_q_learning_solves_two_armed_contextual_bandit():
    # state-independent rewards: action 0 pays 1, others 0
    ag = QAgent(alpha=0.2, gamma=0.0, eps=0.2, seed=0)
    s = (0, 1)
    for _ in range(500):
        a = ag.act(s, list(range(NG + 1)))
        ag.update(s, a, 1.0 if a == 0 else 0.0, s, True)
    assert ag.act(s, list(range(NG + 1)), greedy=True) == 0


def test_env_rewards_and_masks():
    steps = 3
    rm = np.full((steps, ALL + 1), 1.0); rm[:, 1] = 0.5        # only group 0 is useful
    T = Tables(pd.bdate_range("2020-01-01", periods=steps), rm, rm.copy(), np.full(steps, ALL),
               np.ones(steps), np.full(steps, 20.0), np.zeros(steps), np.full(steps, 21))
    env = FeatureSelectionEnv(T, np.arange(steps))
    env.reset(mask=1)
    _, r, _, e = env.step(KEEP)
    assert e == 0.5 and r == pytest.approx(100 * 0.5 - 0.05)
    assert popcount(ALL) == NG and len(mask_features(ALL)) == len(FEATURES)
    assert regime(30, 0.1) == 5 and regime(12, -0.1) == 0


def test_rewards_are_learned_only_after_the_next_decision():
    steps = 3
    rm = np.full((steps, ALL + 1), 1.0); rm[:, 1] = 0.5
    T = Tables(pd.bdate_range("2020-01-01", periods=steps), rm, rm.copy(), np.full(steps, ALL),
               np.ones(steps), np.full(steps, 20.0), np.zeros(steps), np.full(steps, 21))
    ag = QAgent(alpha=1.0, gamma=0.0, eps=0.0, seed=0)
    seen, act = [], ag.act

    def spy(s, valid, greedy=False):
        seen.append(sum(float(np.abs(v).sum()) for v in ag.Q.values()))
        return act(s, valid, greedy)
    ag.act = spy
    run_episode(FeatureSelectionEnv(T, np.arange(steps)), ag)
    assert seen[0] == 0 and seen[1] == 0 and seen[2] > 0     # decision 3 knows the reward of decision 1 only


def test_purged_blocks_drop_decisions_whose_labels_cross_a_boundary():
    from lab7_rl_features.env import purged_blocks
    dates = pd.to_datetime(["2014-11-03", "2014-12-15", "2015-02-02", "2018-12-10", "2019-01-15"])
    ends = pd.to_datetime(["2014-12-10", "2015-01-22", "2015-03-11", "2019-01-17", "2019-02-21"])
    tr, va, te = purged_blocks(dates, ends)
    assert list(tr) == [0] and list(va) == [2] and list(te) == [4]
