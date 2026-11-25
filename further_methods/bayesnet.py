"""Discrete Bayesian networks (Han et al., p. 393-397; Scutari 2010).

A Bayesian network is a directed acyclic graph whose nodes are variables and a conditional probability table (CPT) of
every node given its parents; the joint distribution is the product of the CPTs, P(x_1, ..., x_n) = prod_i P(x_i |
parents(x_i)). The structure is learned by hill climbing on the BIC score (or AIC, with a smaller penalty): starting from the empty graph, the single
addition, deletion or reversal of an arc that increases the score most is made, while the graph stays acyclic and no
arc of the blacklist (expert knowledge) is used. The score of a node is its maximized log-likelihood given its parents
minus (log n / 2) times the number of free parameters of its CPT, so the score of a graph is the sum over the nodes.
Inference: a conditional probability is obtained by summing the joint distribution over the unobserved variables.
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd


def node_score(D: pd.DataFrame, node: str, parents: tuple, levels: dict, penalty: str = "bic") -> float:
    n = len(D)
    r = len(levels[node])
    q = int(np.prod([len(levels[p]) for p in parents])) if parents else 1
    if parents:
        counts = D.groupby(list(parents) + [node], observed=True).size()
        tot = counts.groupby(level=list(range(len(parents))), observed=True).transform("sum")
    else:
        counts = D.groupby(node, observed=True).size()
        tot = pd.Series(n, index=counts.index)
    ll = float((counts * np.log(counts / tot)).sum())
    return ll - (0.5 * np.log(n) if penalty == "bic" else 1.0) * q * (r - 1)


def _acyclic(parents: dict) -> bool:
    seen, stack = set(), set()

    def visit(v):
        if v in stack:
            return False
        if v in seen:
            return True
        stack.add(v)
        ok = all(visit(p) for p in parents[v])
        stack.discard(v); seen.add(v)
        return ok
    return all(visit(v) for v in parents)


def hill_climb(D: pd.DataFrame, blacklist: set = frozenset(), max_parents: int = 2, max_iter: int = 200, penalty: str = "bic"):
    """Returns ({node: tuple(parents)}, score, list of moves)."""
    nodes = list(D.columns)
    levels = {c: sorted(D[c].unique()) for c in nodes}
    parents = {v: () for v in nodes}
    cache = {}

    def sc(v, ps):
        key = (v, tuple(sorted(ps)))
        if key not in cache:
            cache[key] = node_score(D, v, key[1], levels, penalty)
        return cache[key]

    total = sum(sc(v, parents[v]) for v in nodes)
    moves = []
    for _ in range(max_iter):
        best = (1e-9, None)
        for a, b in itertools.permutations(nodes, 2):           # arc a -> b
            if a in parents[b]:
                cand = [("delete", {b: tuple(p for p in parents[b] if p != a)})]
                if (b, a) not in blacklist and len(parents[a]) < max_parents:
                    cand.append(("reverse", {b: tuple(p for p in parents[b] if p != a), a: parents[a] + (b,)}))
            elif b not in parents[a] and (a, b) not in blacklist and len(parents[b]) < max_parents:
                cand = [("add", {b: parents[b] + (a,)})]
            else:
                continue
            for kind, change in cand:
                new = dict(parents); new.update(change)
                if kind != "delete" and not _acyclic(new):
                    continue
                delta = sum(sc(v, new[v]) - sc(v, parents[v]) for v in change)
                if delta > best[0]:
                    best = (delta, (kind, a, b, new))
        if best[1] is None:
            break
        kind, a, b, parents = best[1]
        total += best[0]
        moves.append((kind, a, b, best[0]))
    return parents, total, moves


def cpts(D: pd.DataFrame, parents: dict, alpha: float = 1.0) -> dict:
    """CPTs with a uniform Dirichlet prior (alpha pseudo-counts per cell): {node: DataFrame indexed by parent values}."""
    levels = {c: sorted(D[c].unique()) for c in D.columns}
    out = {}
    for v, ps in parents.items():
        if ps:
            idx = (pd.MultiIndex.from_product([levels[p] for p in ps], names=list(ps)) if len(ps) > 1
                   else pd.Index(levels[ps[0]], name=ps[0]))         # one parent: a plain index, so .loc gives a number
            c = D.groupby(list(ps) if len(ps) > 1 else ps[0], observed=True)[v].value_counts().unstack(fill_value=0)
            c = c.reindex(index=idx, columns=levels[v], fill_value=0) + alpha
        else:
            c = D[v].value_counts().reindex(levels[v], fill_value=0).to_frame().T + alpha
        out[v] = c.div(c.sum(axis=1), axis=0)
    return out


def joint(cpt: dict, parents: dict) -> pd.Series:
    """The joint distribution as a Series over all value combinations (small networks only)."""
    nodes = list(parents)
    levels = {v: list(cpt[v].columns) for v in nodes}
    rows = []
    for combo in itertools.product(*[levels[v] for v in nodes]):
        x = dict(zip(nodes, combo))
        p = 1.0
        for v in nodes:
            t = cpt[v]
            p *= float(t.loc[tuple(x[q] for q in parents[v]) if len(parents[v]) > 1 else x[parents[v][0]], x[v]]
                       if parents[v] else t.iloc[0][x[v]])
        rows.append((*combo, p))
    J = pd.DataFrame(rows, columns=nodes + ["p"])
    return J.set_index(nodes)["p"]


def query(J: pd.Series, target: str, evidence: dict) -> pd.Series:
    """P(target | evidence) by summing the joint distribution (inference by enumeration)."""
    df = J.reset_index()
    for k, v in evidence.items():
        df = df[df[k] == v]
    s = df.groupby(target)["p"].sum()
    return s / s.sum()
