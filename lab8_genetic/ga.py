"""Lab 8. A compact genetic algorithm (minimisation).

Genome: a list of integer genes; gene i takes values 0 .. cardinality[i]-1.
Operators (Han, Kamber & Pei, Sect. 9.6.1; Mitchell 1996):
  * selection   - tournament of size k (selection pressure grows with k);
  * crossover   - uniform crossover with probability p_cx;
  * mutation    - every gene is redrawn with probability p_mut;
  * elitism     - the best ``elite`` individuals are copied unchanged.
Fitness values are cached, so a genome is never evaluated twice.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np


@dataclass
class GAResult:
    best: tuple
    best_fit: float
    history: list = field(default_factory=list)      # (gen, best, mean, median, diversity, evaluations)
    cache: dict = field(default_factory=dict)
    evaluations: int = 0


def diversity(pop) -> float:
    P = np.array(pop)
    n = len(P)
    d = [(P[i] != P[j]).mean() for i in range(n) for j in range(i + 1, n)]
    return float(np.mean(d)) if d else 0.0


class GeneticAlgorithm:
    def __init__(self, cardinality, fitness: Callable | None = None, batch_fitness: Callable | None = None,
                 pop_size=20, generations=10, tournament=3, p_cx=0.9, p_mut=None, elite=2,
                 seed=0, repair: Callable | None = None, init: list | None = None):
        self.card = list(cardinality)
        self.fitness, self.batch_fitness = fitness, batch_fitness
        self.pop_size, self.generations, self.k = pop_size, generations, tournament
        self.p_cx = p_cx
        self.p_mut = p_mut if p_mut is not None else 1.0 / len(self.card)
        self.elite, self.repair, self.init = elite, repair, init
        self.rng = np.random.default_rng(seed)

    def random_genome(self):
        g = tuple(int(self.rng.integers(c)) for c in self.card)
        return self.repair(g) if self.repair else g

    def _evaluate(self, pop, res: GAResult):
        todo = list(dict.fromkeys(g for g in pop if g not in res.cache))
        if todo:
            vals = self.batch_fitness(todo) if self.batch_fitness else [self.fitness(g) for g in todo]
            for g, v in zip(todo, vals):
                res.cache[g] = float(v)
            res.evaluations += len(todo)
        return np.array([res.cache[g] for g in pop])

    def _tournament(self, pop, fit):
        idx = self.rng.choice(len(pop), self.k, replace=False)
        return pop[idx[np.argmin(fit[idx])]]

    def _crossover(self, a, b):
        if self.rng.uniform() > self.p_cx:
            return a, b
        m = self.rng.uniform(size=len(a)) < 0.5
        c1 = tuple(x if mi else y for x, y, mi in zip(a, b, m))
        c2 = tuple(y if mi else x for x, y, mi in zip(a, b, m))
        return c1, c2

    def _mutate(self, g):
        g = list(g)
        for i, c in enumerate(self.card):
            if self.rng.uniform() < self.p_mut:
                g[i] = int(self.rng.integers(c))
        g = tuple(g)
        return self.repair(g) if self.repair else g

    def run(self) -> GAResult:
        res = GAResult(best=None, best_fit=np.inf)
        pop = list(self.init or [])[: self.pop_size]
        while len(pop) < self.pop_size:
            pop.append(self.random_genome())
        for gen in range(self.generations + 1):
            fit = self._evaluate(pop, res)
            order = np.argsort(fit)
            if fit[order[0]] < res.best_fit:
                res.best, res.best_fit = pop[order[0]], float(fit[order[0]])
            res.history.append((gen, float(fit.min()), float(fit.mean()), float(np.median(fit)), diversity(pop),
                                res.evaluations))
            if gen == self.generations:
                break
            new = [pop[i] for i in order[: self.elite]]
            while len(new) < self.pop_size:
                c1, c2 = self._crossover(self._tournament(pop, fit), self._tournament(pop, fit))
                new.append(self._mutate(c1))
                if len(new) < self.pop_size:
                    new.append(self._mutate(c2))
            pop = new
        return res
