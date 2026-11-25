"""Lab 8 (iteration 2) tests: budget accounting of the search methods, PBT replay, the GA engine."""
import numpy as np
import pytest
import torch

from lab8_genetic import search as S


class Toy:
    """A small linear problem with the interface of ``search.Data``."""

    def __init__(self, n=900, seed=0):
        rng = np.random.default_rng(seed)
        X = rng.normal(size=(n, 9)).astype(np.float32)
        y = (0.7 * X[:, 1] + 0.1 * rng.normal(size=n)).astype(np.float32)
        self.X, self.y = torch.as_tensor(X), torch.as_tensor(y)
        r = np.arange(19, n)
        self.rows = {"train": r[:500], "valA": r[500:650], "valB": r[650:780], "test": r[780:]}


@pytest.fixture(autouse=True)
def small_space(monkeypatch):
    monkeypatch.setattr(S, "SPACE", {"lookback": [1, 5], "width": [8, 16], "layers": [1, 2], "dropout": [0.0, 0.1],
                                     "lr": [3e-3, 1e-2], "wd": [0.0, 1e-4], "batch": [64, 128]})
    monkeypatch.setattr(S, "KEYS", list(S.SPACE))
    monkeypatch.setattr(S, "CARD", [2] * 7)
    torch.set_num_threads(1)


RUNS = {"random": lambda: S.random_search(Toy(), 0, budget=40),
        "random2": lambda: S.random_search(Toy(), 0, n_seeds=2, budget=40),
        "ga": lambda: S.genetic(Toy(), 0, budget=40, pop=4),
        "tpe": lambda: S.tpe(Toy(), 0, budget=40),
        "hyperband": lambda: S.hyperband(Toy(), 0, budget=40, R=9),
        "pbt": lambda: S.pbt(Toy(), 0, budget=40, pop=4, ready=2)}


@pytest.mark.parametrize("method", list(RUNS))
def test_methods_respect_the_budget(method):
    res = RUNS[method]()
    assert res["used"] <= 40 and res["winner"] is not None
    assert all(e[3] <= 40 for e in res["evals"])
    assert res["valA"] < 0.3                                  # the toy target has variance about 0.5


def test_pbt_winner_can_be_replayed():
    res = S.pbt(Toy(), 1, budget=24, pop=4, ready=2)
    assert len(res["schedule"]) == 24 // 4                    # every member trains budget / pop epochs
    r = S.replay(Toy(), res, seed=5)
    assert r.epochs == len(res["schedule"]) and r.cfg["lr"] == res["schedule"][-1]["lr"]


def test_run_keeps_the_best_epoch():
    r = S.Run({"lookback": 1, "width": 8, "layers": 1, "dropout": 0.0, "lr": 1e-2, "wd": 0.0, "batch": 64}, 0, Toy())
    r.fit(6, patience=None)
    assert r.mse("valA") == pytest.approx(min(r.hist), rel=1e-5)


def test_encode_decode_roundtrip():
    g = (1, 0, 1, 0, 1, 0, 1)
    assert S.encode(S.decode(g)) == g
