"""FP-growth: frequent itemsets without candidate generation (Han et al., p. 257-259).

The transactions are compressed into an FP-tree: the frequent items of each transaction, sorted by descending support,
form a path from the root, and shared prefixes share nodes that count the transactions passing through them; a header
table links the nodes of each item. Mining starts from the least frequent item: its conditional pattern base (the prefix
paths of its nodes with their counts) is a smaller database, from which a conditional FP-tree is built and mined
recursively; every frequent itemset is found once, with its count, and no candidate itemset is ever generated.
"""
from __future__ import annotations

from collections import defaultdict

import pandas as pd


class _Node:
    __slots__ = ("item", "count", "parent", "children", "link")

    def __init__(self, item, parent):
        self.item, self.count, self.parent, self.children, self.link = item, 0, parent, {}, None


def build_tree(transactions: list[tuple[list, int]], min_count: int):
    """FP-tree of weighted transactions [(items, count)]. Returns (root, header {item: first node}, item counts, nodes)."""
    counts = defaultdict(int)
    for items, w in transactions:
        for i in set(items):
            counts[i] += w
    counts = {i: c for i, c in counts.items() if c >= min_count}
    rank = {i: r for r, i in enumerate(sorted(counts, key=lambda i: (-counts[i], str(i))))}
    root, header, last, nodes = _Node(None, None), {}, {}, 0
    for items, w in transactions:
        node = root
        for i in sorted({i for i in items if i in rank}, key=rank.get):
            child = node.children.get(i)
            if child is None:
                child = node.children[i] = _Node(i, node)
                nodes += 1
                if i in last:
                    last[i].link = child
                else:
                    header[i] = child
                last[i] = child
            child.count += w
            node = child
    return root, header, counts, nodes


def fpgrowth(B: pd.DataFrame, min_support: float, max_len: int | None = None) -> tuple[dict, dict]:
    """Frequent itemsets of a boolean item matrix, in the format of lab5_fuzzy.apriori ({frozenset: support}), and
    statistics (number of nodes of the FP-tree, number of conditional trees built)."""
    n = len(B)
    min_count = min_support * n
    names = list(B.columns)
    M = B.to_numpy()
    tx = [([names[j] for j in row.nonzero()[0]], 1) for row in M]
    out, stats = {}, {"tree nodes": 0, "conditional trees": 0}

    def mine(transactions, suffix):
        root, header, counts, nodes = build_tree(transactions, min_count)
        if not suffix:
            stats["tree nodes"] = nodes
        else:
            stats["conditional trees"] += 1
        for item in sorted(counts, key=lambda i: (counts[i], str(i))):   # least frequent first
            itemset = suffix | {item}
            out[frozenset(itemset)] = counts[item] / n
            if max_len is not None and len(itemset) >= max_len:
                continue
            base, node = [], header[item]
            while node is not None:                                       # conditional pattern base of the item
                path, p = [], node.parent
                while p is not None and p.item is not None:
                    path.append(p.item); p = p.parent
                if path:
                    base.append((path, node.count))
                node = node.link
            if base:
                mine(base, itemset)

    mine(tx, frozenset())
    return out, stats
