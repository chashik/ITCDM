"""Lab 8 tests: genetic operators and search behaviour."""
import numpy as np

from lab8_genetic.ga import GeneticAlgorithm, diversity
from lab8_genetic.run_features import ridge_rmse
from lab8_genetic.run_lstm import CARD, KEYS, decode, repair


def test_onemax_is_solved():
    n = 30
    ga = GeneticAlgorithm([2] * n, fitness=lambda g: n - sum(g), pop_size=40, generations=60, seed=1)
    res = ga.run()
    assert res.best_fit == 0


def test_elitism_makes_best_monotone():
    rng = np.random.default_rng(0)
    w = rng.normal(size=12)
    ga = GeneticAlgorithm([4] * 12, fitness=lambda g: float(np.dot(w, g)), pop_size=16, generations=20, seed=2)
    h = np.array(ga.run().history)
    assert np.all(np.diff(h[:, 1]) <= 1e-12)


def test_cache_prevents_reevaluation():
    calls = []
    def f(g):
        calls.append(g); return sum(g)
    res = GeneticAlgorithm([2] * 4, fitness=f, pop_size=10, generations=10, seed=3).run()
    assert len(calls) == len(set(calls)) == res.evaluations <= 16


def test_diversity_bounds():
    assert diversity([(0, 0), (0, 0)]) == 0.0
    assert diversity([(0, 1), (1, 0)]) == 1.0


def test_lstm_genome_decoding_and_repair():
    g = repair(tuple([0] * len(CARD)))
    cfg = decode(g)
    assert cfg["cell"] == "rnn" and len(cfg["inputs"]) == 1 and set(KEYS) <= set(cfg)


def test_ridge_rmse_uses_only_selected_and_available():
    rng = np.random.default_rng(4)
    Zt = rng.normal(size=(300, 3)); Zt -= Zt.mean(0)          # contract: standardised training block
    yt = 2 * Zt[:, 0] + 0.01 * rng.normal(size=300)
    Ze = rng.normal(size=(50, 3)); ye = 2 * Ze[:, 0]
    ok = np.array([True, True, True])
    assert ridge_rmse((Zt, yt, Ze, ye, ok), [1, 0, 0]) < 0.05
    assert ridge_rmse((Zt, yt, Ze, ye, ok), [0, 1, 0]) > 1.0
    ok0 = np.array([False, True, True])           # feature 0 unavailable -> cannot help
    assert ridge_rmse((Zt, yt, Ze, ye, ok0), [1, 0, 0]) > 1.0


def test_wide_networks_go_to_the_gpu_only_if_there_is_one():
    from lab8_genetic.run_lstm import GPU_HIDDEN, device_for
    assert device_for({"hidden": GPU_HIDDEN}, gpu=True) == "cuda"
    assert device_for({"hidden": GPU_HIDDEN // 2}, gpu=True) == "cpu"
    assert device_for({"hidden": 2 * GPU_HIDDEN}, gpu=False) == "cpu"


def test_gpu_and_cpu_jobs_do_not_overlap(monkeypatch):
    """Without DMFIN_OVERLAP=1 every GPU job has finished before the first CPU job is submitted."""
    from lab8_genetic import run_lstm as R
    events = []

    class Fut:
        def __init__(self, v, d):
            self.v, self.d = v, d

        def result(self):
            events.append(("done", self.d))
            return self.v

    class Pool:
        def __init__(self, d):
            self.d = d

        def submit(self, fn, args):
            events.append(("submit", self.d))
            return Fut(fn(args), self.d)

    t = R.Trainer.__new__(R.Trainer)
    t.gpu, t.n_gpu, t.pools = True, 0, {"cpu": Pool("cpu"), "cuda": Pool("cuda")}
    cfgs = [{"hidden": h, "layers": 1, "lookback": 20} for h in (8, 128, 16, 64)]
    monkeypatch.delenv("DMFIN_OVERLAP", raising=False)
    assert t.map(lambda a: a[0], cfgs, [(i,) for i in range(4)]) == [0, 1, 2, 3]
    first_cpu = events.index(("submit", "cpu"))
    assert events[:first_cpu].count(("done", "cuda")) == 2 and ("submit", "cuda") not in events[first_cpu:]
