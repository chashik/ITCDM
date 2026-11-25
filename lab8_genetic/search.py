"""Lab 8 (iteration 2). Hyperparameter search methods compared at an equal budget.

The tuned model is a global multilayer perceptron for the 5-day log volatility of the 62
series of the panel (``dmfin.panel``); its inputs are the last L days of the nine panel
features.  The budget of every search is a number of training epochs, so that methods
which stop poor configurations early (Hyperband) or reuse weights (PBT) are charged only
for the work they do.

Blocks: training 2000-2014; validation A 2015-2016 (search criterion and early stopping);
validation B 2017-2018 (nested check of the selected configuration); test 2019-2026.

Methods: random search (one or two seeds per configuration), a genetic algorithm (``ga.py``),
TPE (Bergstra et al. 2011, via Optuna), Hyperband (Li et al. 2018) and population-based
training (Jaderberg et al. 2017).
"""
from __future__ import annotations

import copy
import math

import numpy as np
import pandas as pd
import torch
from torch import nn

from .ga import GeneticAlgorithm

SPACE = {"lookback": [1, 5, 10, 20], "width": [16, 32, 64, 128], "layers": [1, 2, 3],
         "dropout": [0.0, 0.1, 0.2, 0.3, 0.5], "lr": [1e-4, 3e-4, 1e-3, 3e-3, 1e-2],
         "wd": [0.0, 1e-5, 1e-4, 1e-3, 1e-2], "batch": [256, 512, 1024, 2048]}
KEYS = list(SPACE)
CARD = [len(v) for v in SPACE.values()]
DEFAULT = {"lookback": 20, "width": 64, "layers": 2, "dropout": 0.1, "lr": 1e-3, "wd": 1e-4, "batch": 1024}
MAX_EPOCHS, PATIENCE, BUDGET = 27, 4, 600
VAL_A_END = "2016-12-31"


def decode(g) -> dict:
    return {k: SPACE[k][i] for k, i in zip(KEYS, g)}


def encode(cfg) -> tuple:
    return tuple(SPACE[k].index(cfg[k]) for k in KEYS)


class Data:
    """Standardized panel features and targets as tensors; row indices of the blocks.

    All blocks use the rows that have 20 days of history, so every lookback is evaluated
    on the same forecasts.
    """

    def __init__(self):
        from lab6_rnn.run_global import prepare
        p, X, y, rows = prepare()
        self.panel = p
        self.X, self.y = torch.as_tensor(X), torch.as_tensor(y)
        d = p["date"].to_numpy()
        va = rows["val"]
        self.rows = {"train": rows["train"], "valA": va[d[va] <= np.datetime64(VAL_A_END)],
                     "valB": va[d[va] > np.datetime64(VAL_A_END)], "test": rows["test"]}
        # purge: the target of the last 5 days of validation A reaches into validation B
        a = self.rows["valA"]
        last = pd.Series(d[a]).groupby(p["series"].to_numpy()[a]).rank(ascending=False).to_numpy() <= 5
        self.rows["valA"] = a[~last]


class MLP(nn.Module):
    def __init__(self, n_in, width, layers, dropout):
        super().__init__()
        mods, d = [], n_in
        for _ in range(layers):
            mods += [nn.Linear(d, width), nn.ReLU(), nn.Dropout(dropout)]
            d = width
        self.body = nn.Sequential(*mods)
        self.out = nn.Linear(d, 1)

    def set_dropout(self, p):
        for m in self.body:
            if isinstance(m, nn.Dropout):
                m.p = p

    def forward(self, x):
        return self.out(self.body(x)).squeeze(-1)


class Run:
    """A resumable training run; the weights of the epoch with the lowest validation-A loss are kept."""

    def __init__(self, cfg: dict, seed: int, data: Data):
        self.cfg, self.data, self.seed = dict(cfg), data, seed
        torch.manual_seed(seed)
        self.offs = torch.arange(-cfg["lookback"] + 1, 1)
        self.net = MLP(9 * cfg["lookback"], cfg["width"], cfg["layers"], cfg["dropout"])
        self.opt = torch.optim.AdamW(self.net.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
        self.gen = torch.Generator().manual_seed(seed)
        self.epochs, self.hist, self.schedule = 0, [], []
        self.best, self.best_state, self.bad = math.inf, None, 0

    def set_hparams(self, lr=None, wd=None, dropout=None, batch=None):
        for k, v in (("lr", lr), ("wd", wd), ("dropout", dropout), ("batch", batch)):
            if v is not None:
                self.cfg[k] = v
        for gr in self.opt.param_groups:
            gr["lr"], gr["weight_decay"] = self.cfg["lr"], self.cfg["wd"]
        self.net.set_dropout(self.cfg["dropout"])

    def _x(self, r):
        return self.data.X[r[:, None] + self.offs].reshape(len(r), -1)

    def train_epoch(self) -> float:
        self.net.train()
        r = torch.as_tensor(self.data.rows["train"])
        perm = r[torch.randperm(len(r), generator=self.gen)]
        bs = self.cfg["batch"]
        for i in range(0, len(perm), bs):
            b = perm[i:i + bs]
            loss = nn.functional.mse_loss(self.net(self._x(b)), self.data.y[b])
            self.opt.zero_grad(); loss.backward(); self.opt.step()
        self.epochs += 1
        self.schedule.append({k: self.cfg[k] for k in ("lr", "wd", "dropout", "batch")})
        v = self.mse("valA", best=False)
        self.hist.append(v)
        if v < self.best:
            self.best, self.bad = v, 0
            self.best_state = copy.deepcopy(self.net.state_dict())
        else:
            self.bad += 1
        return v

    def fit(self, max_epochs: int, patience: int | None = PATIENCE) -> int:
        """Train at most ``max_epochs`` more epochs; returns the number of epochs trained."""
        n = 0
        while n < max_epochs and (patience is None or self.bad < patience):
            self.train_epoch(); n += 1
        return n

    @torch.no_grad()
    def predict(self, block: str, best: bool = True) -> np.ndarray:
        if best and self.best_state is not None:
            cur = copy.deepcopy(self.net.state_dict()); self.net.load_state_dict(self.best_state)
        self.net.eval()
        r = torch.as_tensor(self.data.rows[block])
        out = torch.cat([self.net(self._x(r[i:i + 8192])) for i in range(0, len(r), 8192)]).numpy()
        if best and self.best_state is not None:
            self.net.load_state_dict(cur)
        return out

    def mse(self, block: str, best: bool = True) -> float:
        y = self.data.y[torch.as_tensor(self.data.rows[block])].numpy()
        return float(np.mean((self.predict(block, best) - y) ** 2))


class Exhausted(Exception):
    pass


class Search:
    """Book-keeping shared by the methods: the budget, every evaluation and the best run so far."""

    def __init__(self, data: Data, budget: int = BUDGET):
        self.data, self.budget, self.used = data, budget, 0
        self.evals = []          # (configuration, validation-A MSE, validation-B MSE, budget used so far)
        self.best_run, self.best = None, math.inf

    @property
    def left(self) -> int:
        return self.budget - self.used

    def record(self, cfg, score, run: Run):
        self.evals.append((dict(cfg), float(score), run.mse("valB"), self.used))
        if score < self.best:
            self.best, self.best_run = float(score), run

    def train(self, cfg, seed, max_epochs=MAX_EPOCHS, patience=PATIENCE) -> Run:
        if self.left <= 0:
            raise Exhausted
        r = Run(cfg, seed, self.data)
        self.used += r.fit(min(max_epochs, self.left), patience)
        return r

    def result(self, method, seed) -> dict:
        b = self.best_run
        return {"method": method, "seed": seed, "evals": self.evals, "used": self.used,
                "winner": dict(b.cfg) if b is not None else None, "init_cfg": getattr(b, "init_cfg", None),
                "schedule": list(b.schedule) if b is not None else None, "valA": self.best,
                "valB": b.mse("valB"), "test": b.mse("test"), "pred_test": b.predict("test")}


def sample(rng) -> dict:
    return {k: v[int(rng.integers(len(v)))] for k, v in SPACE.items()}


def random_search(data, seed, n_seeds=1, budget=BUDGET):
    S, rng = Search(data, budget), np.random.default_rng(seed)
    try:
        while True:
            cfg = sample(rng)
            runs = [S.train(cfg, 1000 * seed + j) for j in range(n_seeds)]
            S.record(cfg, np.mean([r.best for r in runs]), min(runs, key=lambda r: r.best))
    except Exhausted:
        pass
    return S.result("random" if n_seeds == 1 else f"random{n_seeds}", seed)


def genetic(data, seed, budget=BUDGET, pop=10):
    S = Search(data, budget)

    def batch(gs):
        out = []
        for g in gs:
            r = S.train(decode(g), 1000 * seed + len(S.evals))
            S.record(decode(g), r.best, r)
            out.append(r.best)
        return out
    ga = GeneticAlgorithm(CARD, batch_fitness=batch, pop_size=pop, generations=1000, tournament=3, p_cx=0.9,
                          p_mut=1.5 / len(CARD), elite=2, seed=seed)
    try:
        ga.run()
    except Exhausted:
        pass
    return S.result("ga", seed)


def tpe(data, seed, budget=BUDGET):
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    S = Search(data, budget)
    study = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=seed, n_startup_trials=10))
    try:
        while True:
            t = study.ask()
            cfg = decode([t.suggest_int(k, 0, len(v) - 1) for k, v in SPACE.items()])
            r = S.train(cfg, 1000 * seed + len(S.evals))
            S.record(cfg, r.best, r)
            study.tell(t, r.best)
    except Exhausted:
        pass
    return S.result("tpe", seed)


def hyperband(data, seed, budget=BUDGET, R=MAX_EPOCHS, eta=3):
    """Brackets of successive halving; a promoted configuration resumes its training."""
    S, rng = Search(data, budget), np.random.default_rng(seed)
    s_max = int(round(math.log(R, eta)))
    try:
        while True:
            for s in range(s_max, -1, -1):
                n = int(math.ceil((s_max + 1) / (s + 1) * eta ** s))
                runs = []
                for _ in range(n):
                    if S.left <= 0:
                        raise Exhausted
                    runs.append(Run(sample(rng), 1000 * seed + len(S.evals) + len(runs), data))
                for i in range(s + 1):
                    r_i = int(round(R * eta ** (i - s)))
                    for run in runs:
                        if S.left <= 0:
                            raise Exhausted
                        S.used += run.fit(min(r_i - run.epochs, S.left), patience=None)
                    if i == s:
                        for run in runs:
                            S.record(run.cfg, run.best, run)
                    runs = sorted(runs, key=lambda r: r.best)[:max(1, len(runs) // eta)]
    except Exhausted:
        pass
    return S.result("hyperband", seed)


def pbt(data, seed, budget=BUDGET, pop=10, ready=3, frac=0.2):
    """Population-based training: exploit (copy a better member) and explore (perturb its hyperparameters)."""
    S, rng = Search(data, budget), np.random.default_rng(seed)
    members = []
    for j in range(pop):
        r = Run(sample(rng), 1000 * seed + j, data)
        r.init_cfg = dict(r.cfg)
        members.append(r)
    per = budget // pop
    k = max(1, int(frac * pop))
    batches = SPACE["batch"]
    for _ in range(0, per, ready):
        for m in members:
            S.used += m.fit(min(ready, per - m.epochs), patience=None)
        order = sorted(range(pop), key=lambda i: members[i].hist[-1])
        for lo in order[-k:]:
            src = members[order[int(rng.integers(k))]]
            new = copy.deepcopy(src)
            new.gen = torch.Generator().manual_seed(int(rng.integers(1 << 31)))
            new.set_hparams(lr=src.cfg["lr"] * rng.choice([0.8, 1.25]), wd=src.cfg["wd"] * rng.choice([0.8, 1.25]),
                            dropout=float(np.clip(src.cfg["dropout"] + rng.choice([-0.05, 0.05]), 0.0, 0.6)),
                            batch=batches[int(np.clip(batches.index(src.cfg["batch"]) + rng.choice([-1, 0, 1]), 0,
                                                      len(batches) - 1))])
            new.bad = 0
            members[lo] = new
    for m in members:
        S.record(m.cfg, m.best, m)
    return S.result("pbt", seed)


def replay(data, res: dict, seed: int) -> Run:
    """Retrain the winner of a search with a new seed: a static configuration, or the PBT schedule."""
    if res["method"] == "pbt":
        r = Run(res["init_cfg"], seed, data)
        for hp in res["schedule"]:
            r.set_hparams(**hp)
            r.train_epoch()
        return r
    r = Run(res["winner"], seed, data)
    r.fit(MAX_EPOCHS, PATIENCE)
    return r


METHODS = {"random": random_search, "random2": lambda d, s: random_search(d, s, n_seeds=2), "ga": genetic,
           "tpe": tpe, "hyperband": hyperband, "pbt": pbt}
